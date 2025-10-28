#!/bin/bash
PROJECT_ID="nt533q13-distributed-ml"
ZONE="asia-southeast1-a"
ZK_NODES="10.10.0.2:2181,10.10.0.3:2181,10.10.0.4:2181"
MASTERS=(spark-master-1 spark-master-2 spark-master-3)
declare -A MASTER_IP_MAP=(
  [spark-master-1]=10.10.0.2
  [spark-master-2]=10.10.0.3
  [spark-master-3]=10.10.0.4
)

echo "🔍 Đang xác định Spark master đang ở trạng thái ALIVE..."
ACTIVE_VM=""
ACTIVE_URL=""

for VM in "${MASTERS[@]}"; do
  echo "   → Kiểm tra $VM ..."
  REMOTE_OUTPUT=$(gcloud compute ssh "$VM" --project="$PROJECT_ID" --zone="$ZONE" \
    --command "python3 - <<'PY'
import json
import urllib.request

try:
    with urllib.request.urlopen('http://localhost:8080/json/', timeout=2) as resp:
        data = json.load(resp)
    if data.get('status') == 'ALIVE':
        print(data.get('url', ''))
except Exception:
    pass
PY" 2>/dev/null)

  ACTIVE_URL=$(echo "$REMOTE_OUTPUT" | grep -Eo 'spark://[^[:space:]]+' | head -n1)
  if [[ -n "$ACTIVE_URL" ]]; then
    ACTIVE_VM="$VM"
    break
  fi
done

if [[ -z "$ACTIVE_VM" ]]; then
  echo "❌ Không xác định được master đang active! Vui lòng kiểm tra lại cụm."
  exit 1
fi

MASTER_HOST=$(echo "$ACTIVE_URL" | sed -n 's@.*//\([^:]*\):.*@\1@p')
if [[ -z "$MASTER_HOST" ]]; then
  MASTER_HOST="$ACTIVE_VM"
fi

if [[ "$MASTER_HOST" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
  MASTER_IP="$MASTER_HOST"
else
  MASTER_IP="${MASTER_IP_MAP[$MASTER_HOST]}"
fi

if [[ -z "$MASTER_IP" ]]; then
  SHORT_HOST="${MASTER_HOST%%.*}"
  MASTER_IP="${MASTER_IP_MAP[$SHORT_HOST]}"
fi

if [[ -z "$MASTER_IP" ]]; then
  MASTER_IP="${MASTER_IP_MAP[$ACTIVE_VM]}"
fi

if [[ -z "$MASTER_IP" ]]; then
  echo "❌ Không xác định được IP tương ứng với master $ACTIVE_VM (thông tin: $ACTIVE_URL)"
  exit 1
fi

echo "✅ Master hiện tại: $ACTIVE_VM ($MASTER_IP)"

echo "🚀 Đang chạy spark-submit trên $ACTIVE_VM ($MASTER_IP)..."

gcloud compute ssh $ACTIVE_VM --project=$PROJECT_ID --zone=$ZONE \
  --command "bash -lc '/opt/spark/bin/spark-submit \
  --master spark://10.10.0.2:7077,10.10.0.3:7077,10.10.0.4:7077 \
  --deploy-mode client \
  --conf spark.deploy.recoveryMode=ZOOKEEPER \
  --conf spark.deploy.zookeeper.url=$ZK_NODES \
  --conf spark.deploy.zookeeper.dir=/spark \
  --conf spark.jars=/opt/spark/jars/gcs-connector-hadoop3-2.2.29.jar \
  --conf spark.hadoop.fs.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem \
  --conf spark.hadoop.fs.AbstractFileSystem.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS \
  --conf spark.hadoop.fs.gs.project.id=$PROJECT_ID \
  --conf spark.executor.memory=12g \
  --conf spark.driver.memory=6g \
  --conf spark.executor.cores=4 \
  gs://nt533q13-spark-data/scripts/train_ml_image_bigdata.py'"
