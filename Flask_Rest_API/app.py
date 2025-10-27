import io
import os
import tempfile
import zipfile
import uuid
import shutil
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Thread, Lock
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image, ImageFile
from flask import Flask, request, render_template, send_file, jsonify, url_for
from pyspark.sql import SparkSession, Row
from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.linalg import Vectors
from pyspark.sql import functions as F

import ray


# ==============================
# 🔧 Cấu hình chung
# ==============================
ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ID = "nt533q13-distributed-ml"
TELCO_MODEL_PATH = "gs://nt533q13-spark-data/models/telco_rf"
DOGCAT_MODEL_PATH = "gs://nt533q13-spark-data/models/dogcat_lr_model"
DOGCAT_LABEL_PATH = "gs://nt533q13-spark-data/models/dogcat_lr_labels"

AVAILABLE_MODELS = [
    {"id": "telco_rf", "name": "Random Forest (Telco Churn)"},
    {"id": "dogcat_lr", "name": "Logistic Regression (Dog vs Cat)"},
]

REQUIRED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges",
]


# ==============================
# 🌟 Flask app
# ==============================
app = Flask(__name__)


# ==============================
# 🐍 Spark lazy init (tiết kiệm RAM + nhanh startup)
# ==============================
_spark = None
_telco_model = None
_dogcat_model = None
_dogcat_labels = None

def get_spark() -> SparkSession:
    global _spark
    if _spark is None:
        _spark = (
            SparkSession.builder
            .appName("MLPredictionAPI")
            # GCS connector — đảm bảo jar đúng đường dẫn
            .config("spark.jars", "/opt/spark/jars/gcs-connector.jar")
            .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
            .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
            .config("spark.hadoop.fs.gs.project.id", PROJECT_ID)
            .getOrCreate()
        )
        print("✅ SparkSession (driver) ready.")
    return _spark


def get_telco_model() -> PipelineModel:
    global _telco_model
    if _telco_model is None:
        spark = get_spark()
        _telco_model = PipelineModel.load(TELCO_MODEL_PATH)
        print("✅ Telco PipelineModel loaded.")
    return _telco_model


def get_dogcat_model() -> LogisticRegressionModel:
    global _dogcat_model
    if _dogcat_model is None:
        spark = get_spark()
        _dogcat_model = LogisticRegressionModel.load(DOGCAT_MODEL_PATH)
        print("✅ DogCat LogisticRegressionModel loaded.")
    return _dogcat_model


def get_dogcat_labels() -> Dict[int, str]:
    global _dogcat_labels
    if _dogcat_labels is None:
        spark = get_spark()
        _dogcat_labels = {
            int(row["index"]): row["label"]
            for row in spark.read.json(DOGCAT_LABEL_PATH).collect()
        }
        print("✅ DogCat labels loaded.")
    return _dogcat_labels


# ==============================
# 🐳 Ray: connect tới head & Actor giữ model/Spark/ResNet ở worker
# ==============================
# Kết nối cluster
ray.init(address="auto")
print(f"✅ Connected to Ray cluster. Nodes: {len(ray.nodes())}")

@ray.remote
class DogCatActor:
    """
    Actor chạy trên Ray worker:
    - Khởi tạo SparkSession (worker-side) 1 lần
    - Nạp ResNet50 (Keras) 1 lần
    - Nạp Spark LogisticRegressionModel + labels 1 lần
    - Cung cấp method process_batch(paths) để xử lý song song
    """
    def __init__(self, project_id: str, model_path: str, label_path: str):
        from tensorflow.keras.applications.resnet50 import ResNet50
        from pyspark.sql import SparkSession

        self.spark = (
            SparkSession.builder
            .appName("DogCatActor")
            .config("spark.jars", "/opt/spark/jars/gcs-connector.jar")
            .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
            .config("spark.hadoop.google.cloud.auth.service.account.enable", "true")
            .config("spark.hadoop.fs.gs.project.id", project_id)
            .getOrCreate()
        )

        # Model
        self.resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        self.cls = LogisticRegressionModel.load(model_path)
        self.labels = {
            int(r["index"]): r["label"]
            for r in self.spark.read.json(label_path).collect()
        }
        print("✅ DogCatActor is ready on worker.")

    def process_batch(self, batch_paths: List[str]) -> List[Dict[str, Any]]:
        from tensorflow.keras.applications.resnet50 import preprocess_input
        from pyspark.sql import Row
        from pyspark.ml.linalg import Vectors

        results = []
        for p in batch_paths:
            try:
                im = Image.open(p).convert("RGB").resize((224, 224))
                arr = np.expand_dims(np.asarray(im).astype("float32"), axis=0)
                arr = preprocess_input(arr)
                feat = self.resnet.predict(arr, verbose=0).flatten()

                df = self.spark.createDataFrame([Row(features_vec=Vectors.dense(feat))])
                pred = self.cls.transform(df).collect()[0]
                idx = int(pred["prediction"])
                prob = float(pred["probability"][idx])

                results.append({
                    "filename": os.path.basename(p),
                    "prediction": self.labels.get(idx, str(idx)),
                    "confidence": f"{prob * 100:.2f}%"
                })
            except Exception as e:
                results.append({"filename": os.path.basename(p), "prediction": "ERROR", "confidence": str(e)})
        return results


# Actor pool (khởi tạo lười, theo nhu cầu)
_actor_pool: List[ray.actor.ActorHandle] = []

def get_or_create_actor_pool(target_workers: int) -> List[ray.actor.ActorHandle]:
    global _actor_pool
    if _actor_pool:
        return _actor_pool

    # Tính toán số actor hợp lý theo CPU cluster
    cpu_total = int(ray.cluster_resources().get("CPU", 1))
    num = max(1, min(target_workers, cpu_total))  # ví dụ: tạo <= tổng CPU
    print(f"🔧 Creating {num} DogCatActor(s) ...")

    _actor_pool = [
        DogCatActor.options(name=f"dogcat_actor_{i}", lifetime="detached").remote(
            PROJECT_ID, DOGCAT_MODEL_PATH, DOGCAT_LABEL_PATH
        )
        for i in range(num)
    ]
    return _actor_pool


# ==============================
# 📦 Batch dog/cat async processing
# ==============================


@dataclass
class BatchJob:
    job_id: str
    zip_path: Path
    batch_size: int
    status: str = "pending"
    result_path: Optional[Path] = None
    error: Optional[str] = None


JOB_STORAGE_DIR = Path(tempfile.gettempdir()) / "dogcat_batch_jobs"
JOB_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

_job_registry: Dict[str, BatchJob] = {}
_job_registry_lock = Lock()
_job_queue: Queue = Queue()
DOGCAT_ACTOR_TARGET = int(os.getenv("DOGCAT_ACTOR_TARGET", "4"))


def _process_dogcat_batch_job(job: BatchJob) -> None:
    job_dir = job.zip_path.parent
    extract_dir = job_dir / "extracted"
    try:
        with zipfile.ZipFile(job.zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        image_paths = [
            os.path.join(root, f)
            for root, _, files in os.walk(extract_dir)
            for f in files
            if Path(f).suffix.lower() in image_exts
        ]
        if not image_paths:
            raise ValueError("No valid image files found in ZIP.")

        pool = get_or_create_actor_pool(target_workers=DOGCAT_ACTOR_TARGET)
        batches = [
            image_paths[i:i + job.batch_size]
            for i in range(0, len(image_paths), job.batch_size)
        ]

        futures = []
        for idx, batch in enumerate(batches):
            actor = pool[idx % len(pool)]
            futures.append(actor.process_batch.remote(batch))

        all_results = []
        for partial in ray.get(futures):
            all_results.extend(partial)

        import pandas as pd

        result_path = job_dir / "dogcat_batch_predictions.csv"
        pd.DataFrame(all_results).to_csv(result_path, index=False)

        with _job_registry_lock:
            job.status = "completed"
            job.result_path = result_path
    except Exception as exc:
        with _job_registry_lock:
            job.status = "failed"
            job.error = str(exc)
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)


def _batch_worker_loop() -> None:
    while True:
        job_id = _job_queue.get()
        try:
            with _job_registry_lock:
                job = _job_registry.get(job_id)
                if job:
                    job.status = "processing"
            if not job:
                continue
            _process_dogcat_batch_job(job)
        finally:
            _job_queue.task_done()


def _start_batch_workers(count: int = 1) -> None:
    for _ in range(count):
        Thread(target=_batch_worker_loop, daemon=True).start()


_start_batch_workers(int(os.getenv("DOGCAT_BATCH_WORKERS", "1")))


# ==============================
# 🧰 Helper
# ==============================
def extract_image_features_resnet(file_storage):
    """(Local) Extract features — chỉ dùng cho route đơn chiếc"""
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    resnet_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    image = Image.open(file_storage.stream).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.asarray(image).astype("float32"), axis=0)
    arr = preprocess_input(arr)
    features = resnet_model.predict(arr, verbose=0)
    return features.flatten().astype("float32")


# ==============================
# 👔 Business logic (đơn chiếc)
# ==============================
def predict_telco_churn(form_data):
    spark = get_spark()
    telco_model = get_telco_model()

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
    if not image_file or image_file.filename == "":
        raise ValueError("Please upload an image.")

    spark = get_spark()
    dogcat_model = get_dogcat_model()
    labels = get_dogcat_labels()

    features = extract_image_features_resnet(image_file)
    df = spark.createDataFrame([Row(features_vec=Vectors.dense(features))])

    prediction = dogcat_model.transform(df).collect()[0]
    pred_idx = int(prediction["prediction"])
    label = labels.get(pred_idx, str(pred_idx))
    prob = float(prediction["probability"][pred_idx])

    return {
        "prediction": label,
        "probability": prob,
        "input_data": {"filename": image_file.filename},
        "model_type": "dogcat_classification",
    }


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

@app.route("/predict/telco", methods=["POST"])
def predict_telco():
    try:
        result = predict_telco_churn(request.form)
        return render_template("result_telco.html", result=result)
    except Exception as e:
        return render_template("result_telco.html", error=str(e))

@app.route("/predict/dogcat", methods=["POST"])
def predict_dogcat():
    try:
        image_file = request.files.get("image")
        result = classify_dog_cat(image_file)
        return render_template("result_dogcat.html", result=result)
    except Exception as e:
        return render_template("result_dogcat.html", error=str(e))


@app.route("/batch_classify_dogcat", methods=["POST"])
def batch_classify_dogcat_route():
    try:
        zip_file = request.files.get("zip_file")
        if not zip_file or zip_file.filename == "":
            return jsonify({"error": "Please upload a ZIP file."}), 400
        if not zip_file.filename.lower().endswith(".zip"):
            return jsonify({"error": "File must be a ZIP archive."}), 400

        batch_size_value = request.form.get("batch_size") or request.args.get("batch_size")
        try:
            batch_size = int(batch_size_value) if batch_size_value else 100
        except ValueError:
            return jsonify({"error": "batch_size must be an integer."}), 400
        if batch_size <= 0:
            return jsonify({"error": "batch_size must be greater than zero."}), 400

        job_id = uuid.uuid4().hex
        job_dir = JOB_STORAGE_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        zip_path = job_dir / "images.zip"
        zip_file.save(zip_path)

        job = BatchJob(job_id=job_id, zip_path=zip_path, batch_size=batch_size)
        with _job_registry_lock:
            _job_registry[job_id] = job
        _job_queue.put(job_id)

        return jsonify({"task_id": job_id, "status": job.status}), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/batch_predict", methods=["POST"])
def batch_predict_telco_csv():
    """Giữ nguyên xử lý Telco theo Spark driver local của Flask"""
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

        spark = get_spark()
        telco_model = get_telco_model()

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


@app.route("/health", methods=["GET"])
def health():
    # Thông tin nhanh: Spark driver đã sẵn sàng chưa, số node Ray
    try:
        spark_ready = get_spark() is not None
    except Exception:
        spark_ready = False
    try:
        ray_nodes = len(ray.nodes())
    except Exception:
        ray_nodes = 0
    return {"ok": True, "spark_ready": spark_ready, "ray_nodes": ray_nodes}, 200


# (Tuỳ chọn) Khởi tạo Spark driver nền để giảm latency request đầu tiên
def warmup_background():
    try:
        get_spark()
        get_telco_model()
        # Không bắt buộc warm ResNet tại Flask (đã có actor), nhưng có thể:
        # from tensorflow.keras.applications.resnet50 import ResNet50
        # _ = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        print("🔥 Warmup done.")
    except Exception as e:
        print(f"Warmup error: {e}")

Thread(target=warmup_background, daemon=True).start()


@app.route("/batch_classify_dogcat/<task_id>", methods=["GET"])
def batch_classify_dogcat_status(task_id: str):
    with _job_registry_lock:
        job = _job_registry.get(task_id)
    if not job:
        return jsonify({"error": "Task not found."}), 404

    response = {"task_id": job.job_id, "status": job.status}
    if job.status == "completed" and job.result_path:
        response["download_url"] = url_for("download_batch_dogcat_result", task_id=job.job_id, _external=True)
    if job.status == "failed" and job.error:
        response["error"] = job.error
    return jsonify(response), 200


@app.route("/batch_classify_dogcat/<task_id>/result", methods=["GET"])
def download_batch_dogcat_result(task_id: str):
    with _job_registry_lock:
        job = _job_registry.get(task_id)
    if not job:
        return jsonify({"error": "Task not found."}), 404
    if job.status != "completed" or not job.result_path:
        return jsonify({"error": "Task not completed."}), 409

    return send_file(
        job.result_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"{task_id}_predictions.csv",
    )


if __name__ == "__main__":
    # Chạy trực tiếp (dev). Prod dùng gunicorn.
    app.run(host="0.0.0.0", port=8080)
