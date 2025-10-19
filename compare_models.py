from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col, when
from pyspark.mllib.evaluation import MulticlassMetrics
import time
import pandas as pd


spark = SparkSession.builder.appName("Compare_RF_GBT").master("local[*]").getOrCreate()
spark.sparkContext.setLogLevel("WARN")


df = spark.read.option("header", True).option("inferSchema", True).csv("/home/group12/distributed-ml/data/telco_churn.csv")
df = df.withColumn("TotalCharges", when(col("TotalCharges") == " ", None)
                   .otherwise(col("TotalCharges")).cast("double"))
df = df.na.drop(subset=["TotalCharges"])
df = df.withColumn("label", (col("Churn") == "Yes").cast("integer"))
replicas = 200
df_big = df
for i in range(replicas - 1):
    df_big = df_big.union(df)


df = df_big


categorical_cols = [
    'gender','Partner','Dependents','PhoneService','MultipleLines',
    'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
    'TechSupport','StreamingTV','StreamingMovies','Contract',
    'PaperlessBilling','PaymentMethod'
]
numeric_cols = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']

indexers = [StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep") for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=c+"_idx", outputCol=c+"_vec") for c in categorical_cols]
assembler = VectorAssembler(inputCols=numeric_cols + [c+"_vec" for c in categorical_cols], outputCol="features")

train, test = df.randomSplit([0.8, 0.2], seed=42)


def evaluate_model(model_name, classifier):
    print(f"\n🚀 Training {model_name}...")
    start = time.time()
    pipeline = Pipeline(stages=indexers + encoders + [assembler, classifier])
    model = pipeline.fit(train)
    train_time = time.time() - start


    predictions = model.transform(test)


    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)


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

output_path = "/home/group12/distributed-ml/data/model_comparison.csv"
pdf.to_csv(output_path, index=False)
print(f"\n💾 Kết quả được lưu tại: {output_path}")

spark.stop()
