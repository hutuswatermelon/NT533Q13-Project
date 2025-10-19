from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, when
from pyspark.mllib.evaluation import MulticlassMetrics
import time

spark = SparkSession.builder.appName("TelcoChurn_GBT").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

df = spark.read.option("header", True).option("inferSchema", True).csv("gs://nt533q13-spark-data/data/telco_customer_churn.csv")

df = df.withColumn("TotalCharges", when(col("TotalCharges") == " ", None)
                   .otherwise(col("TotalCharges")).cast("double"))
df = df.na.drop(subset=["TotalCharges"])
df = df.withColumn("label", (col("Churn") == "Yes").cast("integer"))

categorical_cols = ['gender','Partner','Dependents','PhoneService','MultipleLines',
                    'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
                    'TechSupport','StreamingTV','StreamingMovies','Contract',
                    'PaperlessBilling','PaymentMethod']
numeric_cols = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']

indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_vec") for c in categorical_cols]
assembler = VectorAssembler(inputCols=numeric_cols + [c+"_vec" for c in categorical_cols], outputCol="features")
train, test = df.randomSplit([0.8, 0.2], seed=42)

gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=100, maxDepth=5, stepSize=0.1)

pipeline = Pipeline(stages=indexers + encoders + [assembler, gbt])

print("🚀 Training Gradient Boosted Tree model...")
start_time = time.time()
model = pipeline.fit(train)
train_time = time.time() - start_time
predictions = model.transform(test)
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"\n✅ Training completed in {train_time:.2f} seconds")
print(f"📈 AUC = {auc:.4f}")

rdd = predictions.select("prediction", "label").rdd.map(tuple)
metrics = MulticlassMetrics(rdd)

accuracy = predictions.filter(col("label") == col("prediction")).count() / predictions.count()
print(f"\n🎯 Accuracy: {accuracy:.2%}")
print(f"Precision (rời đi): {metrics.precision(1):.2f}")
print(f"Recall (rời đi): {metrics.recall(1):.2f}")
print(f"F1-score (rời đi): {metrics.fMeasure(1):.2f}")

model_path = "gs://nt533q13-spark-data/models/telco_gbt"
model.write().overwrite().save(model_path)
print(f"\n💾 Model saved to {model_path}")

spark.stop()
