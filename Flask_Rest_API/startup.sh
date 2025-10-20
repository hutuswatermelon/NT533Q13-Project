#!/bin/bash

# === Cập nhật hệ thống và cài đặt gói cần thiết ===
sudo apt update && sudo apt install -y python3-pip python3-venv git default-jdk subversion

# === Tạo thư mục làm việc (nếu chưa có) ===
cd /home/$USER

# === Tải thư mục Flask từ GitHub ===
svn export https://github.com/hutuswatermelon/NT533Q13-Project/trunk/Flask

cd Flask

# === Tạo môi trường ảo và cài dependencies ===
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# === Chạy Flask API ngầm (background) ===
nohup python3 app.py > /home/$USER/flask.log 2>&1 &
