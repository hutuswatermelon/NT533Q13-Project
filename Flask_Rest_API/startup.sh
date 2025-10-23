#!/bin/bash
# Cập nhật hệ thống & cài Python + Git + Gunicorn
sudo apt update && sudo apt install -y python3-pip python3-venv git default-jdk subversion

# Lấy thư mục Flask từ GitHub (dùng svn export)
cd /home/$USER
svn export https://github.com/hutuswatermelon/NT533Q13-Project/trunk/Flask_Rest_API

# Cài đặt Python packages toàn cục
cd Flask_Rest_API
sudo pip3 install --upgrade pip
sudo pip3 install -r requirements.txt
sudo pip3 install gunicorn

# Chạy Flask app với Gunicorn trên cổng 80
sudo nohup gunicorn --bind 0.0.0.0:80 app:app > flask.log 2>&1 &

# Thông báo
echo "Log: tail -f /home/$USER/Flask_Rest_API/flask.log"
