from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, split, element_at, udf
from pyspark.sql.types import ArrayType, FloatType
import io, os
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

def _open_image_from_any(content, origin_path):
    if content is not None:
        b = _to_bytes(content)
        if b:
            try:
                return Image.open(io.BytesIO(b)).convert("RGB").resize((224, 224))
            except Exception:
                pass
    local_path = origin_path.replace("file://", "") if origin_path else None
    if local_path:
        try:
            return Image.open(local_path).convert("RGB").resize((224, 224))
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
print("🧩 Đã trích xuất đặc trưng ResNet50.")

# ====== 7. Chuyển features -> vector MLlib ======
to_vector_udf = udf(lambda arr: Vectors.dense(arr), VectorUDT())
feature_vector_df = feature_df.withColumn("features_vec", to_vector_udf(col("features")))

# ====== 8. Huấn luyện Logistic Regression ======
print("🔧 Bắt đầu huấn luyện mô hình Logistic Regression...")

indexer = StringIndexer(inputCol="label", outputCol="labelIndex")
data_indexed = indexer.fit(feature_vector_df).transform(feature_vector_df)

train_df, test_df = data_indexed.randomSplit([0.8, 0.2], seed=42)

lr = LogisticRegression(featuresCol="features_vec", labelCol="labelIndex", maxIter=50)
lr_model = lr.fit(train_df)

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
with open("train_result.txt", "w") as f:
    f.write(f"🎯 Độ chính xác mô hình trên tập test: {accuracy:.4f}\n")
print("📁 Kết quả đã được lưu vào: train_result.txt")

# ====== 10. Lưu mô hình ======
# os.makedirs("models", exist_ok=True)
# lr_model.write().overwrite().save("models/dogcat_lr_model")
lr_model.write().overwrite().save("gs://nt533q13-spark-data/models/dogcat_lr_model")
print("✅ Mô hình đã lưu tại: gs://nt533q13-spark-data/models/dogcat_lr_model")

spark.stop()

