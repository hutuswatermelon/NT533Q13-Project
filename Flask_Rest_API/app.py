import io
import os
import uuid
import tempfile
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional

import numpy as np
from PIL import Image, ImageFile
from flask import Flask, request, render_template, send_file, jsonify, url_for
from google.cloud import storage
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.linalg import Vectors

# ==============================
# ⚙️ Config
# ==============================
PROJECT_ID = "nt533q13-distributed-ml"

# Bucket cho API (lưu code job + dữ liệu job)
API_BUCKET = "nt533q13-api-data"

# Spark Master REST
SPARK_MASTER_REST = "http://10.10.0.2:6066/v1/submissions/create"

# Spark job script path trong GCS
SPARK_JOB_SCRIPT = "gs://nt533q13-api-data/code/classify_images_job.py"

# Model & labels trong GCS (để Spark job dùng)
TELCO_MODEL_PATH = "gs://nt533q13-spark-data/models/telco_rf"
DOGCAT_MODEL_PATH = "gs://nt533q13-spark-data/models/dogcat_lr_model"
DOGCAT_LABEL_PATH = "gs://nt533q13-spark-data/models/dogcat_lr_labels"

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)
storage_client = storage.Client(project=PROJECT_ID)

# ==============================
# 🔥 SparkSession (cho 3 route cũ)
# ==============================
spark = (
    SparkSession.builder
    .appName("MLPredictionAPI")
    .config("spark.jars", "/opt/spark/jars/gcs-connector.jar")
    .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
    .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
    .config("spark.hadoop.fs.gs.project.id", PROJECT_ID)
    .getOrCreate()
)

# Load models & labels (cho 3 route cũ)
telco_model = PipelineModel.load(TELCO_MODEL_PATH)
dogcat_model = LogisticRegressionModel.load(DOGCAT_MODEL_PATH)
resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
dogcat_labels = {
    int(row["index"]): row["label"]
    for row in spark.read.json(DOGCAT_LABEL_PATH).collect()
}

REQUIRED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges",
]

AVAILABLE_MODELS = [
    {"id": "telco_rf", "name": "Random Forest (Telco Churn)"},
    {"id": "dogcat_lr", "name": "Logistic Regression (Dog vs Cat)"},
]

# ==============================
# 🔧 Helpers (cũ)
# ==============================
def extract_image_features(file_storage):
    """Trích xuất features từ ảnh sử dụng ResNet50 (cho route 1 image)."""
    image = Image.open(file_storage.stream).convert("RGB").resize((224, 224))
    array = np.asarray(image).astype("float32")
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    features = resnet_model.predict(array, verbose=0)
    return features.flatten().astype("float32")


def predict_telco_churn(form_data):
    """Dự đoán churn (đơn chiếc) — GIỮ NGUYÊN."""
    data = {
        "customerID": form_data.get("customerID"),
        "gender": form_data.get("gender"),
        "SeniorCitizen": int(form_data.get("SeniorCitizen")),
        "Partner": form_data.get("Partner"),
        "Dependents": form_data.get("Dependents"),
        "tenure": int(form_data.get("tenure")),
        "PhoneService": form_data.get("PhoneService"),
        "MultipleLines": form_data.get("MultipleLines"),
        "InternetService": form_data.get("InternetService"),
        "OnlineSecurity": form_data.get("OnlineSecurity"),
        "OnlineBackup": form_data.get("OnlineBackup"),
        "DeviceProtection": form_data.get("DeviceProtection"),
        "TechSupport": form_data.get("TechSupport"),
        "StreamingTV": form_data.get("StreamingTV"),
        "StreamingMovies": form_data.get("StreamingMovies"),
        "Contract": form_data.get("Contract"),
        "PaperlessBilling": form_data.get("PaperlessBilling"),
        "PaymentMethod": form_data.get("PaymentMethod"),
        "MonthlyCharges": float(form_data.get("MonthlyCharges")),
        "TotalCharges": float(form_data.get("TotalCharges")),
    }
    data_for_model = {k: v for k, v in data.items() if k != "customerID"}
    df = spark.createDataFrame([data_for_model])
    prediction = telco_model.transform(df).collect()[0]
    label = int(prediction["prediction"])
    prob = float(prediction["probability"][1])
    return {
        "prediction": label,
        "probability": prob,
        "input_data": data,
        "model_type": "telco_churn",
    }


def classify_dog_cat(image_file):
    """Phân loại 1 ảnh dog/cat — GIỮ NGUYÊN."""
    if not image_file or image_file.filename == "":
        raise ValueError("Please upload an image.")
    features = extract_image_features(image_file)
    df = spark.createDataFrame([Row(features_vec=Vectors.dense(features))])
    prediction = dogcat_model.transform(df).collect()[0]
    pred_idx = int(prediction["prediction"])
    label = dogcat_labels.get(pred_idx, str(pred_idx))
    prob = float(prediction["probability"][pred_idx])
    return {
        "prediction": label,
        "probability": prob,
        "input_data": {"filename": image_file.filename},
        "model_type": "dogcat_classification",
    }

# ==============================
# ☁️ GCS utils (mới cho batch image)
# ==============================
def upload_to_gcs(local_path: str, dest_blob: str) -> str:
    bucket = storage_client.bucket(API_BUCKET)
    blob = bucket.blob(dest_blob)
    blob.upload_from_filename(local_path)
    blob.make_private()
    return f"gs://{API_BUCKET}/{dest_blob}"


def generate_signed_url(gcs_uri: str, expiration=3600) -> str:
    bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.generate_signed_url(expiration=expiration)

# ==============================
# 🚀 Spark REST submit (mới cho batch image)
# ==============================
import requests

def submit_spark_job(job_id: str, zip_uri: str, model_path: str, label_path: str, output_dir: str):
    payload = {
        "action": "CreateSubmissionRequest",
        "appResource": SPARK_JOB_SCRIPT,
        "clientSparkVersion": "3.5.7",
        "mainClass": "org.apache.spark.deploy.PythonRunner",
        "appArgs": [
            "--zip-uri", zip_uri,
            "--lr-model-path", model_path,
            "--labels-json", label_path,
            "--output-csv", output_dir
        ],
        "sparkProperties": {
            # ---- Tên app
            "spark.app.name": f"dogcat-{job_id}",
            "spark.submit.deployMode": "cluster",

            # ---- Standalone HA: nhiều master
            "spark.master": "spark://10.10.0.2:7077,10.10.0.3:7077,10.10.0.4:7077",

            # ---- HA khôi phục qua ZooKeeper
            "spark.deploy.recoveryMode": "ZOOKEEPER",
            "spark.deploy.zookeeper.url": "10.10.0.2:2181,10.10.0.3:2181,10.10.0.4:2181",
            "spark.deploy.zookeeper.dir": "/spark",

            # ---- GCS connector (đủ cả FS và AbstractFS)
            "spark.hadoop.fs.gs.impl": "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem",
            "spark.hadoop.fs.AbstractFileSystem.gs.impl": "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS",
            "spark.hadoop.google.cloud.auth.service.account.enable": "true",
            "spark.hadoop.fs.gs.project.id": PROJECT_ID,

            # ---- Tài nguyên executor (tuỳ chỉnh theo cluster của bạn)
            "spark.executor.instances": "4",
            "spark.executor.cores": "2",
            "spark.executor.memory": "6g",

            # ---- Tuỳ chọn: Dynamic Allocation (yêu cầu external shuffle service đang chạy)
            "spark.dynamicAllocation.enabled": "true",
            # BẬT nếu bạn đã chạy external shuffle: 
            "spark.shuffle.service.enabled": "true",

            # (tuỳ chọn) Speculation để né task chậm:
            "spark.speculation": "true",
            "spark.speculation.quantile": "0.75",
            "spark.speculation.multiplier": "1.5"
        }
    }
    r = requests.post(SPARK_MASTER_REST, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


# ==============================
# 🧱 Batch-image state (mới)
# ==============================
@dataclass
class BatchJob:
    id: str
    status: str
    submission_id: Optional[str]
    input_gs: Optional[str]
    output_gs: Optional[str]
    error: Optional[str]

job_registry: Dict[str, BatchJob] = {}
registry_lock = Lock()

# ==============================
# 🌐 Routes
# ==============================
@app.route("/", methods=["GET"])
def index():
    selected_model = request.args.get("model", AVAILABLE_MODELS[0]["id"])
    return render_template(
        "index.html",
        models=AVAILABLE_MODELS,
        selected_model=selected_model,
        required_columns=REQUIRED_COLUMNS,
    )

@app.route("/predict/telco", methods=["POST"], endpoint="predict_telco")
def predict_telco_route():
    """GIỮ NGUYÊN."""
    try:
        result = predict_telco_churn(request.form)
        return render_template("result_telco.html", result=result)
    except Exception as e:
        return render_template("result_telco.html", error=str(e))

@app.route("/predict/dogcat", methods=["POST"])
def predict_dogcat_route():
    """GIỮ NGUYÊN."""
    try:
        image_file = request.files.get("image")
        result = classify_dog_cat(image_file)
        return render_template("result_dogcat.html", result=result)
    except Exception as e:
        return render_template("result_dogcat.html", error=str(e))

@app.route("/batch_predict", methods=["POST"])
def batch_predict_telco_csv():
    """GIỮ NGUYÊN — Batch Telco từ CSV (xử lý tại API bằng Spark driver)."""
    uploaded_file = request.files.get("file")
    if not uploaded_file or uploaded_file.filename == "":
        return render_template("result_telco.html", error="No CSV file provided.")
    if not uploaded_file.filename.lower().endswith(".csv"):
        return render_template("result_telco.html", error="File must be a CSV.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            uploaded_file.save(tmp.name)
            tmp_path = tmp.name

        df = spark.read.option("header", "true").csv(tmp_path)
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            return render_template("result_telco.html", error=f"Missing columns: {', '.join(missing_cols)}")

        df = df.select(*REQUIRED_COLUMNS)
        df = df.withColumn("SeniorCitizen", F.col("SeniorCitizen").cast("int"))
        df = df.withColumn("tenure", F.col("tenure").cast("int"))
        df = df.withColumn("MonthlyCharges", F.col("MonthlyCharges").cast("double"))
        df = df.withColumn(
            "TotalCharges",
            F.when(F.trim(F.col("TotalCharges")) == "", None).otherwise(F.col("TotalCharges"))
        )
        df = df.withColumn("TotalCharges", F.col("TotalCharges").cast("double"))
        df = df.fillna({"SeniorCitizen": 0, "tenure": 0, "MonthlyCharges": 0.0, "TotalCharges": 0.0})

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
            download_name="telco_predictions.csv",
        )
    except Exception as e:
        return render_template("result_telco.html", error=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ==============================
# 🌟 Batch image (MỚI) → Spark mapPartitions
# ==============================
@app.route("/batch_classify_dogcat", methods=["POST"])
def batch_classify_dogcat_submit():
    """
    Upload ZIP (>25MB ok) → GCS → Submit Spark job
    Spark sẽ sinh duy nhất: gs://<bucket>/jobs/<id>/results/preds.csv
    """
    try:
        zip_file = request.files.get("zip_file")
        if not zip_file or zip_file.filename == "" or not zip_file.filename.lower().endswith(".zip"):
            return jsonify({"error": "Please upload a .zip file via field 'zip_file'."}), 400

        job_id = uuid.uuid4().hex
        tmp_zip = tempfile.mktemp(suffix=".zip")
        zip_file.save(tmp_zip)

        input_uri = upload_to_gcs(tmp_zip, f"jobs/{job_id}/input.zip")
        output_dir = f"gs://{API_BUCKET}/jobs/{job_id}/results"

        resp = submit_spark_job(
            job_id=job_id,
            zip_uri=input_uri,
            model_path=DOGCAT_MODEL_PATH,
            label_path=DOGCAT_LABEL_PATH,
            output_dir=output_dir
        )
        submission_id = resp.get("submissionId")

        with registry_lock:
            job_registry[job_id] = BatchJob(
                id=job_id,
                status="submitted",
                submission_id=submission_id,
                input_gs=input_uri,
                output_gs=output_dir,
                error=None,
            )

        return jsonify({"task_id": job_id, "submission_id": submission_id, "status": "submitted"}), 202
    finally:
        try:
            if 'tmp_zip' in locals() and os.path.exists(tmp_zip):
                os.remove(tmp_zip)
        except Exception:
            pass

@app.route("/batch_classify_dogcat/<job_id>/status", methods=["GET"])
def batch_classify_dogcat_status(job_id):
    """Poll kết quả: khi có preds.csv sẽ trả signed URL."""
    with registry_lock:
        job = job_registry.get(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    bucket = storage_client.bucket(API_BUCKET)
    blob = bucket.blob(f"jobs/{job_id}/results/preds.csv")
    if blob.exists():
        signed = generate_signed_url(f"gs://{API_BUCKET}/jobs/{job_id}/results/preds.csv")
        job.status = "completed"
        return jsonify({"task_id": job_id, "status": "completed", "result_url": signed}), 200

    return jsonify({"task_id": job_id, "status": job.status}), 200

# ==============================
# 🩺 Health
# ==============================
@app.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    # Prod dùng gunicorn, đây là dev
    app.run(host="0.0.0.0", port=8080)
