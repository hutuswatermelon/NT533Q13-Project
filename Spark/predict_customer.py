from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

spark = SparkSession.builder.appName("PredictTelcoChurn").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

model_path = "gs://nt533q13-spark-data/models/telco_rf"
model = PipelineModel.load(model_path)
print(f"✅ Model loaded from {model_path}")

data = [
    {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 845.5,
    }
]

df_new = spark.createDataFrame(data)

try:
    predictions = model.transform(df_new)
    result = predictions.select(
        "gender", "Contract", "MonthlyCharges", "TotalCharges", "prediction", "probability"
    )
    result.show(truncate=False)
    for row in result.collect():
        prob_leave = row['probability'][1]
        decision = "🚪 Rời đi" if row['prediction'] == 1.0 else "🏠 Ở lại"
        print(f"\nKhách hàng: {row['gender']}, Hợp đồng: {row['Contract']}")
        print(f"💰 Phí hàng tháng: {row['MonthlyCharges']} | Tổng chi phí: {row['TotalCharges']}")
        print(f"🔮 Dự đoán: {decision} (xác suất rời đi = {prob_leave:.2%})")
finally:
    spark.stop()

