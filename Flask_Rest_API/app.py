from flask import Flask, request, jsonify, render_template
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
    # ID của project GCP
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

# =========================
# Giao diện nhập dữ liệu
# =========================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# ====================================
# 🚀 API predict
# ====================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Lấy dữ liệu từ form
        data = {
            "gender": request.form.get("gender"),
            "SeniorCitizen": int(request.form.get("SeniorCitizen")),
            "Partner": request.form.get("Partner"),
            "Dependents": request.form.get("Dependents"),
            "tenure": int(request.form.get("tenure")),
            "PhoneService": request.form.get("PhoneService"),
            "InternetService": request.form.get("InternetService"),
            "MonthlyCharges": float(request.form.get("MonthlyCharges")),
            "TotalCharges": float(request.form.get("TotalCharges"))
        }

        # Chuyển thành DataFrame Spark
        df = spark.createDataFrame([data])

        # Dự đoán
        prediction = model.transform(df).collect()[0]
        label = int(prediction["prediction"])
        prob = float(prediction["probability"][1])

        return render_template("result.html",
                               prediction=label,
                               probability=prob,
                               data=data)
    except Exception as e:
        return render_template("result.html", error=str(e))

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
