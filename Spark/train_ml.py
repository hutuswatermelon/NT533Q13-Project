from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, when  

spark = SparkSession.builder.appName("TelcoChurnPrediction").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

df = spark.read.option("header", True).option("inferSchema", True).csv("gs://nt533q13-spark-data/data/telco_customer_churn.csv")

df = df.withColumn(
    "TotalCharges",
    when(col("TotalCharges") == " ", None).otherwise(col("TotalCharges")).cast("double")
)
df = df.na.drop(subset=["TotalCharges"])
df = df.withColumn("label", (df["Churn"] == "Yes").cast("integer"))

categorical_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]
numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_vec") for c in categorical_cols]
assembler = VectorAssembler(
    inputCols=numeric_cols + [c+"_vec" for c in categorical_cols],
    outputCol="features"
)
rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxDepth=8)
pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

train, test = df.randomSplit([0.8, 0.2], seed=42)

model = pipeline.fit(train)
predictions = model.transform(test)

evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"AUC = {auc:.4f}")

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = acc_eval.evaluate(predictions)
print(f"🎯 Accuracy = {accuracy:.4f}")

from pyspark.sql.functions import col
total = predictions.count()
stay = predictions.filter(col("prediction") == 0.0).count()
leave = predictions.filter(col("prediction") == 1.0).count()
print(f"🏠 Ở lại: {stay} khách hàng")
print(f"🚪 Rời đi: {leave} khách hàng")
print(f"Tổng cộng: {total} khách hàng")

model.write().overwrite().save("gs://nt533q13-spark-data/models/telco_rf")
print("✅ Model saved to gs://nt533q13-spark-data/models/telco_rf")

print("\n📋 Kết quả chi tiết (10 khách hàng đầu tiên):")
predictions.select("customerID", "gender", "Contract", "MonthlyCharges", 
                   "Churn", "prediction", "probability").show(10, truncate=False)

spark.stop()
