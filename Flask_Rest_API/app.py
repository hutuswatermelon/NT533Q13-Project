from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

app = Flask(__name__)

# Khởi tạo Spark session, dùng ADC (Application Default Credentials)
spark = SparkSession.builder \
    .appName("ChurnPredictionAPI") \
    .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
    .getOrCreate()

# 📌 Đường dẫn tới mô hình đã train và lưu trên GCS
MODEL_PATH = "gs://nt533q13-spark-data/models/telco_rf"

print("🔹 Đang load mô hình...")
model = PipelineModel.load(MODEL_PATH)
print("✅ Load mô hình thành công!")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert JSON sang Spark DataFrame
        df = spark.createDataFrame([data])

        # Dự đoán
        prediction = model.transform(df).collect()[0]
        label = prediction["prediction"]
        prob = prediction["probability"][1]  # xác suất churn

        return jsonify({
            "churn_prediction": int(label),
            "churn_probability": float(prob)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
