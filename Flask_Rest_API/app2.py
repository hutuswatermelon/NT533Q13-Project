import io
import os
import tempfile
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image, ImageFile
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from pyspark.sql import SparkSession, Row
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.linalg import Vectors
from pyspark.sql import functions as F
import zipfile
from pathlib import Path
from celery import Celery
import redis

app = Flask(__name__)

# ====================================
# ⚙️ Khởi tạo SparkSession với GCS connector
# ====================================
spark = (
    SparkSession.builder
    .appName("MLPredictionAPI")
    .config("spark.jars", "/opt/spark/jars/gcs-connector.jar")
    .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
    .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
    .config("spark.hadoop.fs.gs.project.id", "nt533q13-distributed-ml")
    .getOrCreate()
)

# ====================================
# 📦 Load models từ GCS
# ====================================
ImageFile.LOAD_TRUNCATED_IMAGES = True
TELCO_MODEL_PATH = "gs://nt533q13-spark-data/models/telco_rf"
DOGCAT_MODEL_PATH = "gs://nt533q13-spark-data/models/dogcat_lr_model"
DOGCAT_LABEL_PATH = "gs://nt533q13-spark-data/models/dogcat_lr_labels"

# Initialize Celery
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# Load models
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

# ====================================
# 🔧 Helper functions
# ====================================
def extract_image_features(file_storage):
    image = Image.open(file_storage.stream).convert("RGB").resize((224, 224))
    array = np.asarray(image).astype("float32")
    array = np.expand_dims(array, axis=0)
    array = preprocess_input(array)
    features = resnet_model.predict(array, verbose=0)
    return features.flatten().astype("float32")

@celery.task(bind=True)
def process_images(zip_path):
    results = []
    tmp_dir = tempfile.mkdtemp()

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(tmp_dir)

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        image_paths = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(tmp_dir)
            for file in files
            if Path(file).suffix.lower() in image_extensions
        ]

        for img_path in image_paths:
            try:
                features = extract_image_features(open(img_path, 'rb'))
                df = spark.createDataFrame([Row(features_vec=Vectors.dense(features))])
                prediction = dogcat_model.transform(df).collect()[0]
                pred_idx = int(prediction["prediction"])
                label = dogcat_labels.get(pred_idx, str(pred_idx))
                prob = float(prediction["probability"][pred_idx])

                results.append({
                    "filename": os.path.basename(img_path),
                    "prediction": label,
                    "confidence": f"{prob * 100:.2f}%",
                })
            except Exception as e:
                results.append({
                    "filename": os.path.basename(img_path),
                    "prediction": "ERROR",
                    "confidence": str(e),
                })

        return results
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        if os.path.exists(tmp_dir):
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

@app.route("/batch_classify_dogcat", methods=["POST"])
def batch_classify_dogcat_route():
    zip_file = request.files.get("zip_file")
    if not zip_file or zip_file.filename == "":
        return jsonify({"error": "Please upload a ZIP file."}), 400
    if not zip_file.filename.lower().endswith(".zip"):
        return jsonify({"error": "File must be a ZIP archive."}), 400

    tmp_zip_path = tempfile.mktemp(suffix=".zip")
    zip_file.save(tmp_zip_path)

    task = process_images.apply_async(args=[tmp_zip_path])
    return jsonify({"task_id": task.id}), 202

@app.route("/task_status/<task_id>", methods=["GET"])
def task_status(task_id):
    task = process_images.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'result': task.result if task.state == 'SUCCESS' else None
        }
    else:
        response = {
            'state': task.state,
            'error': str(task.info),  # this will be the exception raised
        }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)