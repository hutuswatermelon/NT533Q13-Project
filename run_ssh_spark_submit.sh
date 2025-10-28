#!/bin/bash
PROJECT_ID="nt533q13-distributed-ml"
ZONE="asia-southeast1-a"
ZK_NODES="10.10.0.2:2181,10.10.0.3:2181,10.10.0.4:2181"

echo "🔍 Đang truy vấn master_status từ ZooKeeper..."
RAW_OUTPUT=$(gcloud compute ssh spark-master-1 --project=$PROJECT_ID --zone=$ZONE \
  --command "sudo /opt/zookeeper/bin/zkCli.sh -server $ZK_NODES get /spark/master_status" 2>/dev/null)

MASTER_IP=$(echo "$RAW_OUTPUT" | grep -Eo '([0-9]{1,3}\.){3}[0-9]{1,3}' | tail -n1)

if [ -z "$MASTER_IP" ]; then
  echo "❌ Không xác định được master đang active!"
  echo "Output từ ZooKeeper:"
  echo "$RAW_OUTPUT"
  exit 1
fi

echo "✅ Master hiện tại: $MASTER_IP"

# Map IP → tên VM
if [[ "$MASTER_IP" == "10.10.0.2" ]]; then
  MASTER_VM="spark-master-1"
elif [[ "$MASTER_IP" == "10.10.0.3" ]]; then
  MASTER_VM="spark-master-2"
elif [[ "$MASTER_IP" == "10.10.0.4" ]]; then
  MASTER_VM="spark-master-3"
else
  echo "⚠️ IP $MASTER_IP không khớp VM nào đã biết!"
  exit 1
fi

echo "🚀 Đang chạy spark-submit trên $MASTER_VM ($MASTER_IP)..."

gcloud compute ssh $MASTER_VM --project=$PROJECT_ID --zone=$ZONE \
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
