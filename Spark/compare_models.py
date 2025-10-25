from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, when
from pyspark.mllib.evaluation import MulticlassMetrics
import time
import pandas as pd
spark = SparkSession.builder.appName("TelcoChurn_GBT").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
spark = SparkSession.builder.appName("Compare_RF_GBT").getOrCreate()
spark.sparkContext.setLogLevel("WARN").option("inferSchema", True).csv("gs://nt533q13-spark-data/data/telco_customer_churn.csv")

df = df.withColumn("TotalCharges", when(col("TotalCharges") == " ", None)
df = spark.read.option("header", True).option("inferSchema", True).csv("gs://nt533q13-spark-data/data/telco_customer_churn.csv")
df = df.withColumn("TotalCharges", when(col("TotalCharges") == " ", None)
                   .otherwise(col("TotalCharges")).cast("double"))))
df = df.na.drop(subset=["TotalCharges"])
df = df.withColumn("label", (col("Churn") == "Yes").cast("integer"))MultipleLines',
replicas = 200      'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
df_big = df         'TechSupport','StreamingTV','StreamingMovies','Contract',
for i in range(replicas - 1):sBilling','PaymentMethod']
    df_big = df_big.union(df)n','tenure','MonthlyCharges','TotalCharges']

indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in categorical_cols]
df = df_big[OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_vec") for c in categorical_cols]
assembler = VectorAssembler(inputCols=numeric_cols + [c+"_vec" for c in categorical_cols], outputCol="features")
train, test = df.randomSplit([0.8, 0.2], seed=42)
categorical_cols = [
    'gender','Partner','Dependents','PhoneService','MultipleLines',er=100, maxDepth=5, stepSize=0.1)
    'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
    'TechSupport','StreamingTV','StreamingMovies','Contract',gbt])
    'PaperlessBilling','PaymentMethod'
]rint("🚀 Training Gradient Boosted Tree model...")
numeric_cols = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']
model = pipeline.fit(train)
indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_vec") for c in categorical_cols]
assembler = VectorAssembler(inputCols=numeric_cols + [c+"_vec" for c in categorical_cols], outputCol="features")
auc = evaluator.evaluate(predictions)
train, test = df.randomSplit([0.8, 0.2], seed=42)} seconds")
print(f"📈 AUC = {auc:.4f}")

def evaluate_model(model_name, classifier):el").rdd.map(tuple)
    print(f"\n🚀 Training {model_name}...")
    start = time.time()
    pipeline = Pipeline(stages=indexers + encoders + [assembler, classifier])edictions.count()
    model = pipeline.fit(train)y:.2%}")
    train_time = time.time() - start.precision(1):.2f}")
print(f"Recall (rời đi): {metrics.recall(1):.2f}")
print(f"F1-score (rời đi): {metrics.fMeasure(1):.2f}")
    predictions = model.transform(test)
model_path = "gs://nt533q13-spark-data/models/telco_gbt"
model.write().overwrite().save(model_path)
    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)
spark.stop()

    accuracy = predictions.filter(col("label") == col("prediction")).count() / predictions.count()


    rdd = predictions.select(
        col("prediction").cast("double"),
        col("label").cast("double")
    ).rdd.map(tuple)

    metrics = MulticlassMetrics(rdd)
    precision = metrics.precision(1.0)
    recall = metrics.recall(1.0)
    f1 = metrics.fMeasure(1.0)


    result = {
        "Model": model_name,
        "AUC": round(auc, 4),
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "TrainTime(s)": round(train_time, 2)
    }

    print(f"✅ Done {model_name}: AUC={auc:.4f}, Accuracy={accuracy:.2%}, Time={train_time:.2f}s")
    return result



rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=100, maxDepth=8)
gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=100, maxDepth=5, stepSize=0.1)


results = []
results.append(evaluate_model("Random Forest", rf))
results.append(evaluate_model("Gradient Boosted Tree", gbt))

pdf = pd.DataFrame(results)
print("\n📊 Kết quả so sánh mô hình:")
print(pdf)

output_path = "gs://nt533q13-spark-data/data/model_comparison.csv"
pdf.to_csv(output_path, index=False)
print(f"\n💾 Kết quả được lưu tại: {output_path}")

spark.stop()
