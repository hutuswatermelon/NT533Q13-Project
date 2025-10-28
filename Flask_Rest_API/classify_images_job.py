# classify_images_job.py
import argparse, os, io, zipfile, json, math, tempfile
from pyspark.sql import SparkSession

def copy_gcs_to_local(sc, gcs_uri: str, local_path: str):
    """Dùng Hadoop FS API (GCS connector) để copy gs:// → local (không cần gsutil)."""
    jsc = sc._jsc
    jvm = sc._jvm
    hconf = jsc.hadoopConfiguration()
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(hconf)
    Path = jvm.org.apache.hadoop.fs.Path
    fs.copyToLocalFile(False, Path(gcs_uri), Path(local_path), True)

def write_single_csv(sc, lines_rdd, output_dir, filename="preds.csv"):
    """Gộp thành MỘT file CSV duy nhất: output_dir/preds.csv"""
    tmp_dir = f"{output_dir}/_tmp_pred"
    header = sc.parallelize(["filename,label,confidence"], 1)
    header.union(lines_rdd).coalesce(1).saveAsTextFile(tmp_dir)

    jvm = sc._jvm
    hconf = sc._jsc.hadoopConfiguration()
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(hconf)
    Path = jvm.org.apache.hadoop.fs.Path

    # tìm part-*
    part_path = None
    for status in fs.listStatus(Path(tmp_dir)):
        name = status.getPath().getName()
        if name.startswith("part-"):
            part_path = status.getPath()
            break
    if part_path is None:
        raise RuntimeError("No part file generated.")

    # tạo thư mục đích nếu chưa có
    fs.mkdirs(Path(output_dir))
    dest = Path(f"{output_dir}/{filename}")
    # xóa nếu đã tồn tại
    if fs.exists(dest):
        fs.delete(dest, True)
    # rename part -> preds.csv
    fs.rename(part_path, dest)
    # dọn thư mục tạm
    fs.delete(Path(tmp_dir), True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-uri", required=True)         # gs://.../jobs/<id>/input.zip
    ap.add_argument("--lr-model-path", required=True)   # gs://.../models/dogcat_lr_model
    ap.add_argument("--labels-json", required=True)     # gs://.../models/dogcat_lr_labels (file hoặc folder json)
    ap.add_argument("--output-csv", required=True)      # gs://.../jobs/<id>/results  (sẽ tạo preds.csv)
    ap.add_argument("--repartition", type=int, default=0)
    args = ap.parse_args()

    spark = (SparkSession.builder.appName("DogCatBatchZIP").getOrCreate())
    sc = spark.sparkContext

    # ===== 1) đọc labels về driver =====
    labels = {}
    try:
        for line in sc.textFile(args.labels_json).collect():
            try:
                obj = json.loads(line)
                labels[int(obj["index"])] = obj["label"]
            except:
                pass
    except Exception:
        pass
    if not labels:
        # thử trường hợp 1 file json
        content = "\n".join(sc.textFile(args.labels_json).collect())
        labels = json.loads(content)
    bc_labels = sc.broadcast(labels)

    # ===== 2) load LR & broadcast weights/bias =====
    from pyspark.ml.classification import LogisticRegressionModel
    lr = LogisticRegressionModel.load(args.lr_model_path)
    w = lr.coefficients.toArray()
    b = float(lr.intercept)
    bc_w = sc.broadcast(w)
    bc_b = sc.broadcast(b)

    # ===== 3) copy ZIP từ GCS về local & broadcast zip bytes =====
    local_zip = "/tmp/input.zip"
    copy_gcs_to_local(sc, args.zip_uri, local_zip)
    with open(local_zip, "rb") as f:
        zip_bytes = f.read()
    bc_zip = sc.broadcast(zip_bytes)

    # enumerate filename trong zip
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        names = [m.filename for m in z.infolist() if not m.is_dir() and os.path.splitext(m.filename)[1].lower() in exts]
    if not names:
        raise RuntimeError("No valid images found in ZIP.")

    num_slices = args.repartition or max(1, len(names)//32)
    rdd = sc.parallelize(names, numSlices=num_slices)

    # ===== 4) mapPartitions: load ResNet50 1 lần/partition, suy luận tất cả ảnh trong partition =====
    def proc_partition(iter_names):
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
        from PIL import Image
        import numpy as np, io, math, zipfile as _zipfile

        resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        out = []
        zf_bytes = bc_zip.value
        with _zipfile.ZipFile(io.BytesIO(zf_bytes), "r") as z:
            for name in iter_names:
                try:
                    with z.open(name) as f:
                        data = f.read()
                    img = Image.open(io.BytesIO(data)).convert("RGB").resize((224, 224))
                    arr = np.expand_dims(np.asarray(img).astype("float32"), axis=0)
                    arr = preprocess_input(arr)
                    feat = resnet.predict(arr, verbose=0).flatten()

                    w = bc_w.value; b = bc_b.value
                    logit = float(np.dot(feat, w) + b)
                    prob = 1.0 / (1.0 + math.exp(-logit))
                    idx = 1 if prob >= 0.5 else 0
                    label = bc_labels.value.get(idx, str(idx))
                    out.append(f"{name},{label},{prob}")
                except Exception:
                    out.append(f"{name},ERROR,0")
        return out

    results = rdd.mapPartitions(proc_partition)

    # ===== 5) ghi MỘT file CSV duy nhất =====
    write_single_csv(sc, results, args.output_csv, filename="preds.csv")

    # cleanup
    try:
        os.remove(local_zip)
    except Exception:
        pass
    spark.stop()

if __name__ == "__main__":
    main()
