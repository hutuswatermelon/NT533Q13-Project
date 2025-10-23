from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

app = Flask(__name__)

# ====================================
# ⚙️ Khởi tạo SparkSession với GCS connector
# ====================================
spark = (
    SparkSession.builder
    .appName("ChurnPredictionAPI")
    # Nạp GCS connector jar
    .config("spark.jars", "/opt/spark/jars/gcs-connector.jar")
    # Cho phép Spark đọc GCS
    .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
    # Sử dụng Application Default Credentials từ metadata server
    .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
    # ID của project GCP — thay bằng đúng project của bạn
    .config("spark.hadoop.fs.gs.project.id", "nt533q13-distributed-ml")
    .getOrCreate()
)

# ====================================
# 📦 Load model từ GCS
# ====================================
MODEL_PATH = "gs://nt533q13-spark-data/models/telco_rf"

print("🔹 Đang load mô hình...")
model = PipelineModel.load(MODEL_PATH)
print("✅ Load mô hình thành công!")

# ====================================
# 🚀 API predict
# ====================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = spark.createDataFrame([data])

        result = model.transform(df).collect()[0]
        return jsonify({
            "churn_prediction": int(result["prediction"]),
            "churn_probability": float(result["probability"][1])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
