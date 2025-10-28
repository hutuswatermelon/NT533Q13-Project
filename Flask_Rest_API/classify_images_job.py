# classify_images_job.py
import io, os, zipfile, argparse, math, json
from pyspark.sql import SparkSession

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-uri", required=True)
    ap.add_argument("--lr-model-path", required=True)
    ap.add_argument("--labels-json", required=True)
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()

    spark = (SparkSession.builder
             .appName("DogCatZipJob")
             .getOrCreate())
    sc = spark.sparkContext

    from pyspark.ml.classification import LogisticRegressionModel
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    from PIL import Image
    import numpy as np
    import io, zipfile, tempfile
    from pyspark import SparkFiles

    # Download zip từ GCS sang local tmp
    tmp_zip = "/tmp/input.zip"
    os.system(f"gsutil cp {args.zip_uri} {tmp_zip}")

    # Mở zip và list ảnh
    zf = zipfile.ZipFile(tmp_zip, "r")
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    members = [m for m in zf.infolist() if os.path.splitext(m.filename)[1].lower() in image_exts]
    if not members:
        raise RuntimeError("No valid images found in zip")

    # Đọc LR model và labels
    lr = LogisticRegressionModel.load(args.lr_model_path)
    labels = {int(r["index"]): r["label"] for r in spark.read.json(args.labels_json).collect()}
    w = lr.coefficients.toArray()
    b = float(lr.intercept)
    bc_w = sc.broadcast(w)
    bc_b = sc.broadcast(b)
    bc_labels = sc.broadcast(labels)

    # Tạo RDD (filename, bytes)
    def gen_images():
        for m in members:
            try:
                content = zf.read(m)
                yield (m.filename, content)
            except Exception:
                yield (m.filename, b"")

    rdd = sc.parallelize(list(gen_images()), numSlices=math.ceil(len(members)/5))

    def process_partition(it):
        from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
        from PIL import Image
        import numpy as np, io, math
        resnet = ResNet50(weights="imagenet", include_top=False, pooling="avg")
        results = []
        for name, data in it:
            if not data:
                results.append(f"{name},ERROR,0")
                continue
            try:
                img = Image.open(io.BytesIO(data)).convert("RGB").resize((224,224))
                arr = np.expand_dims(np.asarray(img).astype("float32"), axis=0)
                arr = preprocess_input(arr)
                feat = resnet.predict(arr, verbose=0).flatten()
                w = bc_w.value; b = bc_b.value
                logit = float(np.dot(feat, w) + b)
                prob = 1/(1+math.exp(-logit))
                idx = 1 if prob>=0.5 else 0
                label = bc_labels.value.get(idx,str(idx))
                results.append(f"{name},{label},{prob}")
            except Exception:
                results.append(f"{name},ERROR,0")
        return results

    header = sc.parallelize(["filename,label,confidence"],1)
    out = header.union(rdd.mapPartitions(process_partition))
    out.saveAsTextFile(args.output_csv)
    spark.stop()

if __name__ == "__main__":
    main()
