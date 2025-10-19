from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler

from pyspark.ml.clustering import KMeans

from pyspark.ml.evaluation import ClusteringEvaluator

from pyspark.sql.functions import col, when

import matplotlib.pyplot as plt

from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("TelcoClustering") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


spark.sparkContext.setLogLevel("WARN")


df = spark.read.option("header", True).option("inferSchema", True).csv("/home/group12/distributed-ml/data/telco_churn.csv")


df = df.withColumn("TotalCharges", when(col("TotalCharges") == " ", None).otherwise(col("TotalCharges")).cast("double"))

df = df.na.drop(subset=["TotalCharges"])


categorical_cols = ['gender','Partner','Dependents','PhoneService','MultipleLines',

                    'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',

                    'TechSupport','StreamingTV','StreamingMovies','Contract',

                    'PaperlessBilling','PaymentMethod']

numeric_cols = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']


indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in categorical_cols]

encoders = [OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_vec") for c in categorical_cols]


assembler = VectorAssembler(inputCols=numeric_cols + [c+"_vec" for c in categorical_cols], outputCol="raw_features")


scaler = StandardScaler(inputCol="raw_features", outputCol="features")


from pyspark.ml import Pipeline

pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])

df_transformed = pipeline.fit(df).transform(df)


k = 5 

kmeans = KMeans(k=k, seed=42, featuresCol="features")

model = kmeans.fit(df_transformed)


predictions = model.transform(df_transformed)

predictions.select("gender", "Contract", "MonthlyCharges", "TotalCharges", "prediction").show(10, truncate=False)

predictions.groupBy("prediction").avg("MonthlyCharges", "TotalCharges").show()

evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)

print(f"\n📈 Silhouette with squared euclidean distance = {silhouette:.4f}")


model.save("/home/group12/distributed-ml/models/kmeans_telco")

print("✅ Mô hình KMeans đã được lưu tại /models/kmeans_telco")


sample = predictions.select("MonthlyCharges", "TotalCharges", "prediction").sample(False, 0.1).toPandas()


plt.figure(figsize=(7,5))

plt.scatter(sample["MonthlyCharges"], sample["TotalCharges"], c=sample["prediction"], cmap="rainbow", alpha=0.6)

plt.xlabel("Monthly Charges")

plt.ylabel("Total Charges")

plt.title(f"KMeans Clustering (k={k}) - Telco Customers")

plt.savefig("/home/group12/distributed-ml/data/kmeans_clusters.png")

print("📊 Biểu đồ cụm đã lưu tại: /data/kmeans_clusters.png")


spark.stop()
