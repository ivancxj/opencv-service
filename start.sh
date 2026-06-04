# docker compose up -d --build
# docker build -t opencv-service:4.13.0 -f Dockerfile .
# docker rmi opencv-service:latest
# docker tag opencv-service:4.13.0 opencv-service:latest

# 如果修改了main.py 文件，请重新构建镜像
# docker compose up -d --build 
docker compose up -d
