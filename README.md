# NT533Q13-Project  
## Distributed Machine Learning System with Apache Spark on Google Cloud

---

## Giới thiệu
### Đồ án
Đây là đồ án môn học **Hệ Tính Toán Phân Bố (NT533.Q13)** – Trường Đại học Công nghệ Thông tin, ĐHQG-HCM.  
Dự án tập trung vào việc **triển khai một hệ thống Machine Learning phân tán** sử dụng **Apache Spark** trên **Google Cloud Platform (GCP)**, nhằm đánh giá các đặc tính quan trọng của hệ thống phân tán như **khả năng mở rộng (scalability)**, **khả năng chịu lỗi (fault tolerance)** và **hiệu năng (performance)**.

Hệ thống cho phép huấn luyện mô hình ML trên dữ liệu lớn, lưu trữ dữ liệu và mô hình trên cloud, đồng thời triển khai mô hình dưới dạng **API** để phục vụ dự đoán.

### Thông tin nhóm
Cáp Hữu Tú – 23521696
Huỳnh Ngọc Ngân Tuyền – 23521753

Giảng viên hướng dẫn: TS. Huỳnh Văn Đặng

---

## Báo cáo
Báo cáo chi tiết được trình bày trong file:
Report - Distributed Computing Systems.pdf

---

## Mục tiêu
- Triển khai cụm **Apache Spark Standalone** đa node trên Google Compute Engine
- Xây dựng pipeline **Machine Learning phân tán** bằng PySpark
- Thử nghiệm và so sánh các thuật toán ML phổ biến
- Đánh giá các đặc tính của hệ thống phân tán: scalability, fault tolerance, performance
- Triển khai mô hình đã huấn luyện dưới dạng **dịch vụ API**

---

## Kiến trúc hệ thống
Hệ thống được triển khai hoàn toàn trên **Google Cloud Platform**, bao gồm:

- **Apache Spark Cluster**
  - 3 Spark Master (High Availability, sử dụng ZooKeeper)
  - N Spark Worker (có thể autoscaling)
- **Google Cloud Storage (GCS)**
  - Lưu trữ dataset, mô hình đã huấn luyện và log
- **Flask API + Load Balancer**
  - Cung cấp dịch vụ dự đoán cho người dùng
- **Ray Cluster**
  - Phân tải và xử lý song song các tác vụ ML nặng trong API
- **VPC, Subnet, Firewall**
  - Đảm bảo an toàn và cô lập mạng

---

## Công nghệ sử dụng
- Apache Spark 3.5.x
- PySpark (pyspark.ml)
- Google Cloud Platform (Compute Engine, Cloud Storage, Load Balancer)
- Apache ZooKeeper
- Python 3
- Flask
- Ray
- TensorFlow / Keras
- ResNet50 (Transfer Learning)

---

## Các bài toán Machine Learning
### 1. Classification – Dự đoán khách hàng rời đi (Churn Prediction)
- RandomForestClassifier  
- GBTClassifier  
- Logistic Regression  
- Dataset: Telco Customer Churn  

### 2. Clustering – Phân nhóm khách hàng
- KMeans  
- Chuẩn hóa dữ liệu với StandardScaler  

### 3. Bài toán mở rộng – Phân loại ảnh Chó/Mèo
- Transfer Learning với ResNet50  
- Trích xuất đặc trưng bằng mapPartitions / Pandas UDF  
- Huấn luyện Logistic Regression trên vector đặc trưng  

---

## Cách chạy (tổng quát)
### 1. Chuẩn bị môi trường
```bash
sudo apt install openjdk-11-jdk python3 python3-pip
pip install pyspark tensorflow flask ray
```
### 2. Huấn luyện mô hình (ví dụ)
```bash
spark-submit train_churn_rf.py
```
### 3. Chạy API
```bash
python api_server.py
```
## Kiểm thử & đánh giá
- Kiểm thử khả năng chịu lỗi bằng cách mô phỏng worker node bị lỗi
- Kiểm thử khả năng mở rộng bằng autoscaling Spark Worker
- Kiểm thử hiệu năng API bằng công cụ load testing (Locust)
## Theo dõi và phân tích thông qua
- Spark UI
- Google Cloud Monitoring
- Các chỉ số latency, throughput và error rate

---

## Hạn chế của hệ thống
Mặc dù hệ thống đã đáp ứng được các mục tiêu chính của đồ án và thể hiện rõ các đặc tính của hệ thống phân tán, vẫn còn tồn tại một số hạn chế như sau:

- **Chi phí vận hành trên Cloud:**  
  Việc chạy cụm Spark nhiều node, API autoscaling và xử lý dữ liệu lớn trên GCP có thể phát sinh chi phí đáng kể nếu triển khai lâu dài hoặc ở quy mô lớn, đặc biệt khi không có cơ chế tự động tắt tài nguyên khi nhàn rỗi.

- **Hiệu năng Transfer Learning phụ thuộc phần cứng:**  
  Bài toán phân loại ảnh sử dụng ResNet50 chạy trên CPU, chưa tận dụng GPU/TPU. Điều này làm thời gian huấn luyện và trích xuất đặc trưng còn chậm đối với tập dữ liệu rất lớn.

- **API chưa tối ưu cho production:**  
  Flask API được xây dựng chủ yếu cho mục đích demo và kiểm thử. Hệ thống chưa triển khai các cơ chế nâng cao như authentication, rate limiting, logging tập trung hay circuit breaker.

- **Giám sát và logging còn cơ bản:**  
  Việc theo dõi hệ thống chủ yếu dựa trên Spark UI và Google Cloud Monitoring, chưa tích hợp các giải pháp logging và observability chuyên sâu như ELK stack hoặc Prometheus + Grafana.

- **Chưa xử lý bài toán streaming real-time:**  
  Hệ thống tập trung vào xử lý batch (offline training). Các kịch bản xử lý dữ liệu streaming real-time với Spark Streaming hoặc Structured Streaming chưa được triển khai.

---

## Ghi chú
Dự án phục vụ mục đích học tập và nghiên cứu
Không khuyến nghị sử dụng trực tiếp cho môi trường production
