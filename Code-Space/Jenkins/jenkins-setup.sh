#!/bin/bash

# Jenkins CI/CD 学习环境快速搭建脚本
# 适用于WSL2 + Docker环境

echo "🚀 开始搭建Jenkins学习环境..."

# 创建必要的目录
mkdir -p html
echo "<h1>欢迎来到CI/CD部署测试页面！</h1><p>这是通过Jenkins自动部署的页面</p>" > html/index.html

# 启动服务
echo "📦 启动Docker容器..."
docker-compose up -d jenkins gitea nginx

echo "⏳ 等待Jenkins启动（大约2-3分钟）..."
sleep 30

# 获取Jenkins初始密码
echo "🔑 获取Jenkins初始管理员密码..."
docker exec jenkins-master cat /var/jenkins_home/secrets/initialAdminPassword

echo "✅ 环境搭建完成！"
echo ""
echo "🌐 访问地址："
echo "  Jenkins:  http://localhost:8080"
echo "  Gitea:    http://localhost:3000"
echo "  部署目标:  http://localhost:8082"
echo ""
echo "📝 下一步操作："
echo "1. 访问 http://localhost:8080 配置Jenkins"
echo "2. 使用上面显示的密码登录"
echo "3. 安装推荐插件"
echo "4. 创建管理员用户"
echo ""
echo "🎯 学习建议："
echo "- 先熟悉Jenkins界面和基本概念"
echo "- 创建第一个简单的构建任务"
echo "- 学习Pipeline脚本编写"
echo "- 实践与Git仓库的集成"