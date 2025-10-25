import io
import os
import tempfile
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image, ImageFile
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from pyspark.sql import SparkSession, Row
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.linalg import Vectors
from pyspark.sql import functions as F

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
ImageFile.LOAD_TRUNCATED_IMAGES = True
TELCO_MODEL_PATH = "gs://nt533q13-spark-data/models/telco_rf"
DOGCAT_MODEL_PATH = "gs://nt533q13-spark-data/models/dogcat_lr_model"
DOGCAT_LABEL_PATH = "gs://nt533q13-spark-data/models/dogcat_lr_labels"
AVAILABLE_MODELS = [
    {"id": "telco_rf", "name": "Random Forest (Telco)"},
    {"id": "dogcat_lr", "name": "Logistic Regression (Dog vs Cat)"},
]
print("🔹 Đang load mô hình...")
telco_model = PipelineModel.load(TELCO_MODEL_PATH)
dogcat_model = LogisticRegressionModel.load(DOGCAT_MODEL_PATH)
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
dogcat_labels = {
    int(row["index"]): row["label"]
    for row in spark.read.json(DOGCAT_LABEL_PATH).collect()
}
print("✅ Load mô hình thành công!")

REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

# =========================
# Giao diện nhập dữ liệu
# =========================
@app.route("/", methods=["GET"])
def index():
    selected_model = request.args.get("model", AVAILABLE_MODELS[0]["id"])
    return render_template(
        "index.html",
        models=AVAILABLE_MODELS,
        selected_model=selected_model,
        required_columns=REQUIRED_COLUMNS,
    )

# ====================================
# 🚀 API predict
# ====================================
@app.route("/predict", methods=["POST"])
def predict():
    model_name = request.form.get("model_name", "telco_rf")
    try:
        if model_name == "telco_rf":
            # Lấy dữ liệu từ form
            data = {
                "customerID": request.form.get("customerID"),
                "gender": request.form.get("gender"),
                "SeniorCitizen": int(request.form.get("SeniorCitizen")),
                "Partner": request.form.get("Partner"),
                "Dependents": request.form.get("Dependents"),
                "tenure": int(request.form.get("tenure")),
                "PhoneService": request.form.get("PhoneService"),
                "MultipleLines": request.form.get("MultipleLines"),
                "InternetService": request.form.get("InternetService"),
                "OnlineSecurity": request.form.get("OnlineSecurity"),
                "OnlineBackup": request.form.get("OnlineBackup"),
                "DeviceProtection": request.form.get("DeviceProtection"),
                "TechSupport": request.form.get("TechSupport"),
                "StreamingTV": request.form.get("StreamingTV"),
                "StreamingMovies": request.form.get("StreamingMovies"),
                "Contract": request.form.get("Contract"),
                "PaperlessBilling": request.form.get("PaperlessBilling"),
                "PaymentMethod": request.form.get("PaymentMethod"),
                "MonthlyCharges": float(request.form.get("MonthlyCharges")),
                "TotalCharges": float(request.form.get("TotalCharges"))
            }
            data_for_model = {k: v for k, v in data.items() if k != "customerID"}
            df = spark.createDataFrame([data_for_model])

            # Dự đoán
            prediction = telco_model.transform(df).collect()[0]
            label = int(prediction["prediction"])
            prob = float(prediction["probability"][1])

            return render_template("result.html",
                                   prediction=label,
                                   probability=prob,
                                   data=data)
        if model_name == "dogcat_lr_model":
            uploaded_image = request.files.get("image")
            if not uploaded_image or uploaded_image.filename == "":
                return render_template("result.html", error="Please upload an image.")
            features = extract_image_features(uploaded_image)
            df = spark.createDataFrame([Row(features_vec=Vectors.dense(features))])
            prediction = dogcat_model.transform(df).collect()[0]
            pred_idx = int(prediction["prediction"])
            label = dogcat_labels.get(pred_idx, str(pred_idx))
            prob_vector = prediction["probability"]
            prob = float(prob_vector[pred_idx])
            return render_template("result.html",
                                   prediction=label,
                                   probability=prob,
                                   data={"model": "dogcat_lr_model", "filename": uploaded_image.filename})
        return render_template("result.html", error="Unsupported model selected.")
    except Exception as e:
        return render_template("result.html", error=str(e))

@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    model_name = request.form.get("model_name", "telco_rf")
    if model_name != "telco_rf":
        return render_template("result.html", error="Batch prediction chỉ hỗ trợ mô hình Telco.")
    uploaded_file = request.files.get("file")
    if not uploaded_file or uploaded_file.filename == "":
        return render_template("result.html", error="No CSV file provided.")
    if not uploaded_file.filename.lower().endswith(".csv"):
        return render_template("result.html", error="File must be a CSV.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            uploaded_file.save(tmp.name)
            tmp_path = tmp.name

        df = spark.read.option("header", "true").csv(tmp_path)
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            return render_template("result.html", error=f"Missing columns: {', '.join(missing_cols)}")

        df = df.select(*REQUIRED_COLUMNS)
        df = df.withColumn("SeniorCitizen", F.col("SeniorCitizen").cast("int"))
        df = df.withColumn("tenure", F.col("tenure").cast("int"))
        df = df.withColumn("MonthlyCharges", F.col("MonthlyCharges").cast("double"))
        df = df.withColumn(
            "TotalCharges",
            F.when(F.trim(F.col("TotalCharges")) == "", None).otherwise(F.col("TotalCharges"))
        )
        df = df.withColumn("TotalCharges", F.col("TotalCharges").cast("double"))
        df = df.fillna(
            {"SeniorCitizen": 0, "tenure": 0, "MonthlyCharges": 0.0, "TotalCharges": 0.0}
        )

        df_with_id = df.withColumn("row_id", F.monotonically_increasing_id())
        features_df = df_with_id.drop("customerID")
        predictions = telco_model.transform(features_df)

        result_df = (
            predictions
            .select("row_id", "prediction")
            .join(df_with_id.select("row_id", "customerID"), on="row_id", how="inner")
            .withColumn("churn", F.when(F.col("prediction") == 1, F.lit("yes")).otherwise(F.lit("no")))
            .select(F.col("customerID").alias("id"), "churn")
        )

        csv_buffer = io.StringIO()
        result_df.toPandas().to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        return send_file(
            io.BytesIO(csv_buffer.getvalue().encode("utf-8")),
            mimetype="text/csv",
            as_attachment=True,
            download_name="result.csv",
        )
    except Exception as e:
        return render_template("result.html", error=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

def extract_image_features(file_storage):
    image = Image.open(file_storage.stream).convert("RGB").resize((224, 224))
    array = np.asarray(image).astype("float32")
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    features = resnet_model.predict(array, verbose=0)
    return features.flatten().astype("float32")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
