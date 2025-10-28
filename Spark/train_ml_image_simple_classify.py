from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, split, element_at, udf
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType
import io
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Cho phép load ảnh JPEG bị lỗi nhẹ
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ====== 1. Khởi tạo Spark ======
spark = (
    SparkSession.builder
    .appName("DogCatFeatureExtraction")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")  # 🔇 Ẩn log INFO dài dòng

# ====== 2. Đọc dữ liệu ảnh ======
DATA_PATH = "gs://nt533q13-spark-data/data/PetImages/*"
df = spark.read.format("image").load(DATA_PATH)
df = df.withColumn("label", element_at(split(col("image.origin"), "/"), -2))

# ====== 3. Tải mô hình ResNet50 ======
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# ====== 4. Hàm tiện ích ======
def _to_bytes(x):
    if x is None:
        return None
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    try:
        return x.tobytes()
    except Exception:
        try:
            return bytes(x)
        except Exception:
            return None

def _open_image_from_any(content, _origin_path):
    if content is not None:
        b = _to_bytes(content)
        if b:
            try:
                return Image.open(io.BytesIO(b)).convert("RGB").resize((224, 224))
            except Exception:
                return None
    return None

# ====== 5. UDF trích xuất đặc trưng ======
@pandas_udf(ArrayType(FloatType()))
def extract_features_udf(content_series: pd.Series, origin_series: pd.Series) -> pd.Series:
    out = []
    for content, origin in zip(content_series, origin_series):
        try:
            img = _open_image_from_any(content, origin)
            if img is None:
                out.append([0.0] * 2048)
                continue
            x = np.asarray(img).astype("float32")
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            feat = resnet_model.predict(x, verbose=0)
            out.append(feat.flatten().astype("float32").tolist())
        except Exception as e:
            print("⚠️ Error khi xử lý ảnh:", e)
            out.append([0.0] * 2048)
    return pd.Series(out)

# ====== 6. Trích xuất đặc trưng ======
feature_df = df.withColumn("features", extract_features_udf(col("image.data"), col("image.origin")))
# Loai bo mau loi khi ResNet khong trich duoc dac trung (vector toan 0)
feature_df = feature_df.filter(F.array_max(col("features")) != F.lit(0.0))
print("🧩 Đã trích xuất đặc trưng ResNet50.")

# ====== 7. Chuyển features -> vector MLlib ======
to_vector_udf = udf(lambda arr: Vectors.dense(arr), VectorUDT())
feature_vector_df = feature_df.withColumn("features_vec", to_vector_udf(col("features")))
feature_vector_df = feature_vector_df.cache()

# ====== 8. Huấn luyện Logistic Regression ======
print("🔧 Bắt đầu huấn luyện mô hình Logistic Regression...")

indexer = StringIndexer(inputCol="label", outputCol="labelIndex")
indexer_model = indexer.fit(feature_vector_df)
data_indexed = indexer_model.transform(feature_vector_df)

# Can bang trong so lop de giam lech class imbalance
class_counts = data_indexed.groupBy("labelIndex").count().collect()
total = sum(row["count"] for row in class_counts)
num_classes = max(len(class_counts), 1)
weight_map = {int(row["labelIndex"]): float(total / (num_classes * row["count"])) for row in class_counts if row["count"]}
weight_udf = udf(lambda idx: weight_map.get(int(idx), 1.0), FloatType())
data_indexed = data_indexed.withColumn("classWeight", weight_udf(col("labelIndex")))

train_df, test_df = data_indexed.randomSplit([0.8, 0.2], seed=42)

lr = LogisticRegression(featuresCol="features_vec", labelCol="labelIndex", weightCol="classWeight", maxIter=50, regParam=0.01, elasticNetParam=0.0)
lr_model = lr.fit(train_df)

if train_df.rdd.isEmpty():
    raise RuntimeError("Không còn dữ liệu hợp lệ sau khi lọc; cần nới điều kiện hoặc kiểm tra nguồn ảnh.")

# ====== 9. Đánh giá ======
predictions = lr_model.transform(test_df)

evaluator = MulticlassClassificationEvaluator(
    labelCol="labelIndex",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)

# In ra kết quả và lưu file
print(f"\n🎯 Độ chính xác mô hình trên tập test: {accuracy:.4f}")
RESULTS_DIR = "gs://nt533q13-spark-data/models/dogcat_lr_metrics"
MODEL_DIR = "gs://nt533q13-spark-data/models/dogcat_lr_model"
LABELS_DIR = "gs://nt533q13-spark-data/models/dogcat_lr_labels"
metrics_df = spark.createDataFrame(
    [(float(accuracy),)],
    ["accuracy"]
)
metrics_df.coalesce(1).write.mode("overwrite").json(RESULTS_DIR)
label_df = spark.createDataFrame(
    [(int(idx), label) for idx, label in enumerate(indexer_model.labels)],
    ["index", "label"]
)
label_df.coalesce(1).write.mode("overwrite").json(LABELS_DIR)
lr_model.write().overwrite().save(MODEL_DIR)
print(f"✅ Mô hình đã lưu tại: {MODEL_DIR}")

spark.stop()

