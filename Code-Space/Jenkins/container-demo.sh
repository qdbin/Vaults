#!/bin/bash

# 容器状态演示脚本
echo "🔍 Docker Compose 容器状态演示"

echo "1️⃣ 启动所有服务"
docker-compose up -d

echo ""
echo "2️⃣ 查看运行的容器（注意：是多个独立容器）"
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "3️⃣ 验证容器隔离性"
echo "Jenkins容器的进程："
docker exec jenkins-master ps aux | head -5

echo ""
echo "Gitea容器的进程："
docker exec gitea ps aux | head -5

echo ""
echo "4️⃣ 验证网络连通性"
echo "Jenkins ping Gitea："
docker exec jenkins-master ping -c 2 gitea

echo ""
echo "5️⃣ 验证文件系统隔离"
echo "Jenkins容器根目录："
docker exec jenkins-master ls / | head -5

echo ""
echo "Gitea容器根目录："
docker exec gitea ls / | head -5

echo ""
echo "6️⃣ 查看网络配置"
docker network ls | grep interview

echo ""
echo "✅ 结论："
echo "- 每个service = 一个独立容器"
echo "- 文件系统完全隔离"
echo "- 通过Docker网络互相通信"
echo "- 统一管理，但各自独立运行"