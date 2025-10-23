#!/bin/bash
set -ex
echo "===== Startup script bắt đầu lúc $(date) ====="

# ===============================
# Cập nhật hệ thống
# ===============================
apt-get update -y
apt-get install -y python3 python3-pip openjdk-11-jdk wget git



# ===============================
# Tải GCS Connector cho Spark
# ===============================
mkdir -p /opt/spark/jars
wget -q https://storage.googleapis.com/hadoop-lib/gcs/gcs-connector-hadoop3-2.2.29.jar -O /opt/spark/jars/gcs-connector.jar

# ===============================
# Clone code Flask API
# ===============================
APP_DIR="/opt/app"
if [ ! -d "$APP_DIR" ]; then
    git clone https://github.com/hutuswatermelon/NT533Q13-Project $APP_DIR
fi

cd $APP_DIR/Flask_Rest_API

# ===============================
# Cài thư viện Python
# ===============================
pip3 install -r requirements.txt

# Cho phép python3 bind port 80
setcap 'cap_net_bind_service=+ep' $(readlink -f $(which python3))

# ===============================
# Tạo systemd service
# ===============================
SERVICE_FILE="/etc/systemd/system/flask.service"

cat <<EOL > $SERVICE_FILE
[Unit]
Description=Gunicorn Flask API Service
After=network.target

[Service]
User=root
WorkingDirectory=$APP_DIR/Flask_Rest_API
Environment="JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64"
Environment="PYSPARK_PYTHON=/usr/bin/python3"
Environment="SPARK_CLASSPATH=/opt/spark/jars/gcs-connector.jar"
Environment="HADOOP_CLASSPATH=/opt/spark/jars/gcs-connector.jar"
ExecStart=/usr/local/bin/gunicorn --workers 1 --bind 0.0.0.0:8080 app:app
Restart=always

[Install]
WantedBy=multi-user.target
EOL

systemctl daemon-reload
systemctl enable flask
systemctl start flask
echo "===== Startup script hoàn tất lúc $(date) ====="