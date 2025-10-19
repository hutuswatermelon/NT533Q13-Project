from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt

# 1️⃣ Khởi tạo Spark
spark = SparkSession.builder \
    .appName("WholesaleCustomersClustering") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 2️⃣ Đọc dữ liệu
df = spark.read.option("header", True).option("inferSchema", True).csv("gs://nt533q13-spark-data/data/wholesale_customers.csv")
print("✅ Dữ liệu đã đọc:")
df.show(5)

# 3️⃣ Kiểm tra các cột có sẵn
print("📋 Các cột:", df.columns)

# 4️⃣ Chọn các cột số để phân cụm
numeric_cols = ['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']

# 5️⃣ Gom cột thành vector đặc trưng
assembler = VectorAssembler(inputCols=numeric_cols, outputCol="raw_features")
assembled = assembler.transform(df)

# 6️⃣ Chuẩn hóa dữ liệu (đưa các đặc trưng về cùng thang đo)
scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
scaled_data = scaler.fit(assembled).transform(assembled)

# 7️⃣ Huấn luyện mô hình KMeans
k = 5  # bạn có thể thử k=3-8 để so sánh
kmeans = KMeans(k=k, seed=42, featuresCol="features")
model = kmeans.fit(scaled_data)

# 8️⃣ Dự đoán cụm
predictions = model.transform(scaled_data)
predictions.select(numeric_cols + ["prediction"]).show(10, truncate=False)

# 9️⃣ Đánh giá chất lượng cụm
evaluator = ClusteringEvaluator(featuresCol="features")
silhouette = evaluator.evaluate(predictions)
print(f"\n📈 Silhouette Score (Euclidean distance) = {silhouette:.4f}")

# 1️⃣0️⃣ Tính trung bình theo cụm để hiểu ý nghĩa từng nhóm
print("\n📊 Trung bình mỗi cụm:")
predictions.groupBy("prediction").avg(*numeric_cols).show()

# 1️⃣1️⃣ Lưu mô hình
model.save("gs://nt533q13-spark-data/models/kmeans_wholesale")
print("✅ Mô hình KMeans đã được lưu tại /models/kmeans_wholesale")

# 1️⃣2️⃣ Vẽ biểu đồ 2 chiều (Milk vs Grocery) để minh họa
sample = predictions.select("Milk", "Grocery", "prediction").toPandas()

plt.figure(figsize=(7,5))
plt.scatter(sample["Milk"], sample["Grocery"], c=sample["prediction"], cmap="rainbow", alpha=0.6)
plt.xlabel("Milk (Annual Spending)")
plt.ylabel("Grocery (Annual Spending)")
plt.title(f"KMeans Clustering (k={k}) - Wholesale Customers")
plt.savefig("gs://nt533q13-spark-data/plots/wholesale_clusters.png")
print("📊 Biểu đồ cụm đã lưu tại: /data/wholesale_clusters.png")

spark.stop()
