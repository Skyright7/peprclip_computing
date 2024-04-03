FROM ubuntu:latest

# 安装必需的软件包
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

## 将当前目录下的文件复制到容器中的/app
#COPY . /app

## 将requirements复制到容器中的/app
COPY ./requirements.txt /app

# 安装项目依赖
RUN pip install -r requirements.txt

# 暴露端口8000
EXPOSE 8000

# 运行Django服务器
CMD ["python3", "/app/manage.py", "runserver", "0.0.0.0:8000"]
