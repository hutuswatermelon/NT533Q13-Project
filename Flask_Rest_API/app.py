import io
import os
import uuid
import tempfile
import zipfile
from pathlib import Path
from threading import Lock, RLock
from dataclasses import dataclass
from typing import Dict, Optional

import requests
from flask import Flask, request, jsonify
from google.cloud import storage
from pyspark.sql import SparkSession, Row, functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegressionModel
from PIL import Image, ImageFile
import numpy as np

# ==============================
# ⚙️ Config
# ==============================
PROJECT_ID = "nt533q13-distributed-ml"
API_BUCKET = "nt533q13-api-data"
SPARK_MASTER_REST = "http://10.10.0.2:6066/v1/submissions/create"
SPARK_JOB_SCRIPT = "gs://nt533q13-api-data/code/classify_images_job.py"

TELCO_MODEL_PATH = "gs://nt533q13-spark-data/models/telco_rf"
DOGCAT_MODEL_PATH = "gs://nt533q13-spark-data/models/dogcat_lr_model"
DOGCAT_LABEL_PATH = "gs://nt533q13-spark-data/models/dogcat_lr_labels"

ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)
storage_client = storage.Client(project=PROJECT_ID)

_spark = None
_telco_model = None
_dogcat_model = None
_dogcat_labels = None
_resnet_model = None
_resnet_lock = RLock()


# ==============================
# 🧩 Spark utils
# ==============================
def get_spark():
    global _spark
    if _spark is None:
        _spark = (
            SparkSession.builder
            .appName("FlaskAPI")
            .config("spark.jars", "/opt/spark/jars/gcs-connector.jar")
            .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
            .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
            .config("spark.hadoop.fs.gs.project.id", PROJECT_ID)
            .getOrCreate()
        )
        print("✅ SparkSession ready.")
    return _spark


def get_telco_model():
    global _telco_model
    if _telco_model is None:
        _telco_model = PipelineModel.load(TELCO_MODEL_PATH)
    return _telco_model


def get_dogcat_model():
    global _dogcat_model
    if _dogcat_model is None:
        _dogcat_model = LogisticRegressionModel.load(DOGCAT_MODEL_PATH)
    return _dogcat_model


def get_dogcat_labels():
    global _dogcat_labels
    if _dogcat_labels is None:
        spark = get_spark()
        _dogcat_labels = {
            int(row["index"]): row["label"]
            for row in spark.read.json(DOGCAT_LABEL_PATH).collect()
        }
    return _dogcat_labels


# ==============================
# 🐶 DogCat single image
# ==============================
def _get_resnet_model():
    from tensorflow.keras.applications.resnet50 import ResNet50

    global _resnet_model
    if _resnet_model is None:
        with _resnet_lock:
            if _resnet_model is None:
                _resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    return _resnet_model


def classify_dog_cat(image_file):
    from tensorflow.keras.applications.resnet50 import preprocess_input

    if not image_file:
        raise ValueError("No image uploaded.")

    resnet = _get_resnet_model()
    img = Image.open(image_file.stream).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.asarray(img).astype("float32"), axis=0)
    arr = preprocess_input(arr)
    with _resnet_lock:
        feat = resnet.predict(arr, verbose=0).flatten()

    spark = get_spark()
    model = get_dogcat_model()
    labels = get_dogcat_labels()
    df = spark.createDataFrame([Row(features_vec=Vectors.dense(feat))])
    pred = model.transform(df).collect()[0]

    idx = int(pred["prediction"])
    prob = float(pred["probability"][idx])
    return {"prediction": labels.get(idx, str(idx)), "confidence": f"{prob*100:.2f}%"}


# ==============================
# 📊 Telco churn single
# ==============================
def predict_telco_churn(form_data):
    spark = get_spark()
    model = get_telco_model()

    try:
        data = {
            "customerID": form_data.get("customerID"),
            "gender": form_data.get("gender"),
            "SeniorCitizen": int(form_data.get("SeniorCitizen", 0)),
            "Partner": form_data.get("Partner"),
            "Dependents": form_data.get("Dependents"),
            "tenure": int(form_data.get("tenure", 0)),
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
            "MonthlyCharges": float(form_data.get("MonthlyCharges", 0.0)),
            "TotalCharges": form_data.get("TotalCharges", "0"),
        }
    except ValueError as exc:
        raise ValueError(f"Invalid numeric input: {exc}")

    total_charges = data["TotalCharges"]
    if isinstance(total_charges, str):
        total_charges = total_charges.strip()
    data["TotalCharges"] = float(total_charges) if total_charges not in (None, "") else 0.0

    df = spark.createDataFrame([data])
    pred = model.transform(df).collect()[0]
    return {"prediction": int(pred["prediction"]), "probability": float(pred["probability"][1])}


# ==============================
# ☁️ Helper: upload to GCS
# ==============================
def upload_to_gcs(file_path: str, dest_blob: str):
    bucket = storage_client.bucket(API_BUCKET)
    blob = bucket.blob(dest_blob)
    blob.upload_from_filename(file_path)
    blob.make_private()
    return f"gs://{API_BUCKET}/{dest_blob}"


def generate_signed_url(gcs_uri: str, expiration=3600):
    bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.generate_signed_url(expiration=expiration)


# ==============================
# 🚀 Spark REST submit
# ==============================
def submit_spark_job(job_id: str, zip_uri: str, model_path: str, label_path: str, output_prefix: str):
    payload = {
        "action": "CreateSubmissionRequest",
        "appResource": SPARK_JOB_SCRIPT,
        "clientSparkVersion": "3.5.7",
        "mainClass": "org.apache.spark.deploy.PythonRunner",
        "appArgs": [
            "--zip-uri", zip_uri,
            "--lr-model-path", model_path,
            "--labels-json", label_path,
            "--output-csv", f"{output_prefix}/preds"
        ],
        "sparkProperties": {
            "spark.app.name": f"dogcat-{job_id}",
            "spark.submit.deployMode": "cluster",
            "spark.master": "spark://10.10.0.2:7077",
            "spark.executor.instances": "4",
            "spark.executor.cores": "2",
            "spark.executor.memory": "6g",
            "spark.dynamicAllocation.enabled": "true",
            "spark.hadoop.fs.gs.impl": "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem",
            "spark.hadoop.google.cloud.auth.service.account.enable": "true",
            "spark.hadoop.fs.gs.project.id": PROJECT_ID
        }
    }
    r = requests.post(SPARK_MASTER_REST, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()



# ==============================
# 🧱 Batch endpoints
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


@app.route("/batch_classify_dogcat", methods=["POST"])
def batch_classify_dogcat():
    """Upload ZIP → GCS → Spark REST job"""
    tmp_path = None
    try:
        zip_file = request.files.get("zip_file")
        if not zip_file:
            return jsonify({"error": "Upload ZIP file required"}), 400

        job_id = uuid.uuid4().hex
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            zip_file.save(tmp.name)
            tmp_path = tmp.name

        # Upload ZIP to GCS
        input_uri = upload_to_gcs(tmp_path, f"jobs/{job_id}/input.zip")
        output_prefix = f"gs://{API_BUCKET}/jobs/{job_id}/results"

        # (Option) unzipper service could expand ZIP → GCS images/ prefix
        # For demo, assume already extracted manually or in Spark code

        job_response = submit_spark_job(
            job_id=job_id,
            zip_uri=input_uri,
            model_path=DOGCAT_MODEL_PATH,
            label_path=DOGCAT_LABEL_PATH,
            output_prefix=output_prefix
        )

        submission_id = job_response.get("submissionId", None)

        with registry_lock:
            job_registry[job_id] = BatchJob(
                id=job_id,
                status="submitted",
                submission_id=submission_id,
                input_gs=input_uri,
                output_gs=output_prefix,
                error=None,
            )

        return jsonify({"task_id": job_id, "submission_id": submission_id, "status": "submitted"}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route("/batch_status/<job_id>", methods=["GET"])
def batch_status(job_id):
    with registry_lock:
        job = job_registry.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    bucket = storage_client.bucket(API_BUCKET)
    prefix = f"jobs/{job_id}/results/preds"
    blobs = list(bucket.list_blobs(prefix=prefix))
    if blobs:
        blob = blobs[0]
        result_uri = f"gs://{API_BUCKET}/{blob.name}"
        signed_url = generate_signed_url(result_uri)
        with registry_lock:
            job.status = "completed"
            job.output_gs = result_uri
            job_registry[job_id] = job
        return jsonify({"task_id": job_id, "status": "completed", "result_url": signed_url}), 200

    return jsonify({"task_id": job_id, "status": job.status}), 200


# ==============================
# 🔬 Small routes
# ==============================
@app.route("/predict/dogcat", methods=["POST"])
def predict_dogcat_route():
    image_file = request.files.get("image")
    try:
        result = classify_dog_cat(image_file)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/telco", methods=["POST"])
def predict_telco_route():
    try:
        result = predict_telco_churn(request.form)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
