# Jenkins CI/CD 完整实战指南

## 1. jenkins-setup.sh 脚本执行详解

### 🤔 脚本在哪里执行？
**答案：在你的本机WSL2环境中执行！**

```bash
# 在WSL2中执行（不是在GitLab上）
cd /mnt/c/Users/hanbin/main/快用/project/interview_guaid
chmod +x jenkins-setup.sh
./jenkins-setup.sh
```

### 🔍 为什么在本机执行？
这个脚本的作用是**搭建学习环境**，不是生产部署脚本：

```bash
#!/bin/bash
# 这是环境初始化脚本，类比：装修房子的准备工作

echo "🚀 开始搭建Jenkins学习环境..."
# 1. 创建网页文件（模拟要部署的应用）
mkdir -p html
echo "<h1>欢迎来到CI/CD部署测试页面！</h1>" > html/index.html

# 2. 启动Docker容器（启动Jenkins、GitLab等服务）
docker-compose up -d jenkins gitea nginx

# 3. 获取Jenkins初始密码（自动化获取配置信息）
docker exec jenkins-master cat /var/jenkins_home/secrets/initialAdminPassword
```

### 📋 执行步骤详解

#### 步骤1：环境检查
```bash
# 确保Docker和Docker Compose已安装
docker --version
docker-compose --version
```

#### 步骤2：执行脚本
```bash
# 给脚本执行权限
chmod +x jenkins-setup.sh

# 执行脚本
./jenkins-setup.sh
```

#### 步骤3：访问服务
```bash
# 脚本执行完成后，你可以访问：
# Jenkins:  http://localhost:8080
# Gitea:    http://localhost:3000  
# 部署目标: http://localhost:8082
```

## 2. Jenkinsfile 配置和使用详解

### 🎯 Jenkinsfile 是什么？
**大白话解释：** Jenkinsfile就像是一个"自动化工作清单"，告诉Jenkins要按什么顺序做什么事情。

### 📝 如何使用Jenkinsfile？

#### 方法1：直接在Jenkins中创建Pipeline项目
```bash
# 1. 访问 http://localhost:8080
# 2. 点击"新建任务"
# 3. 选择"Pipeline"类型
# 4. 在Pipeline配置中，选择"Pipeline script"
# 5. 把Jenkinsfile的内容复制粘贴进去
```

#### 方法2：从Git仓库读取Jenkinsfile（推荐）
```bash
# 1. 把Jenkinsfile放到Git仓库根目录
# 2. 在Jenkins中创建Pipeline项目
# 3. 选择"Pipeline script from SCM"
# 4. 配置Git仓库地址
# 5. Jenkins会自动读取Jenkinsfile
```

### 🔧 Jenkinsfile 核心配置解析

```groovy
pipeline {
    // 指定在哪个Agent上运行（any表示任意可用的Agent）
    agent any
    
    // 环境变量定义
    environment {
        PROJECT_NAME = 'interview-demo'  // 项目名称
        DEPLOY_ENV = 'development'       // 部署环境
    }
    
    // 构建阶段定义
    stages {
        stage('代码检出') {
            steps {
                // 从Git仓库拉取代码
                checkout scm
            }
        }
        
        stage('构建应用') {
            steps {
                // 执行构建命令
                sh 'echo "开始构建..."'
                sh 'mkdir -p dist'
                sh 'echo "<h1>Hello CI/CD!</h1>" > dist/index.html'
            }
        }
        
        stage('部署应用') {
            steps {
                // 部署到目标服务器
                sh 'cp dist/* /var/www/html/'
            }
        }
    }
}
```

## 3. Jenkins 主节点和从节点配置

### 🏗️ 架构说明
```
Jenkins Master (主节点)
├── 管理界面和调度
├── 存储配置和构建历史
└── 分发任务给Agent

Jenkins Agent (从节点)
├── 执行具体的构建任务
├── 可以是不同的操作系统
└── 可以有不同的工具环境
```

### 🔧 配置从节点步骤

#### 步骤1：在Master上添加节点
```bash
# 1. 访问Jenkins管理界面
# 2. 系统管理 -> 节点管理
# 3. 新建节点
# 4. 配置节点信息：
#    - 节点名称：agent-01
#    - 远程工作目录：/home/jenkins
#    - 启动方式：通过SSH启动
```

#### 步骤2：配置Agent机器
```bash
# 在Agent机器上创建jenkins用户
sudo useradd -m jenkins
sudo mkdir -p /home/jenkins/.ssh

# 配置SSH密钥认证
ssh-keygen -t rsa -b 4096
# 把公钥添加到Agent机器的authorized_keys
```

#### 步骤3：测试连接
```bash
# Jenkins会自动连接Agent并显示状态
# 绿色圆点 = 连接成功
# 红色叉号 = 连接失败
```

## 4. Pipeline vs Job vs Build 概念解析

### 📊 概念对比表

| 概念 | 定义 | 类比 | 示例 |
|------|------|------|------|
| **Job** | Jenkins中的一个任务项目 | 工厂里的一条生产线 | "构建网站项目" |
| **Pipeline** | Job的一种高级类型 | 生产线的详细工艺流程 | 代码检出→测试→构建→部署 |
| **Build** | Job的一次具体执行 | 生产线的一次生产过程 | "第25次构建" |

### 🔄 它们的关系
```
Job (任务)
├── Build #1 (第1次执行)
├── Build #2 (第2次执行)  
└── Build #3 (第3次执行)

Pipeline Job (流水线任务)
├── Build #1
│   ├── Stage: 代码检出
│   ├── Stage: 单元测试
│   ├── Stage: 构建应用
│   └── Stage: 部署应用
└── Build #2
    ├── Stage: 代码检出
    └── ...
```

### 💡 实际操作示例

#### 创建传统Job
```bash
# 1. 新建任务 -> 自由风格项目
# 2. 配置源码管理（Git仓库）
# 3. 构建触发器（定时构建）
# 4. 构建步骤（执行shell命令）
```

#### 创建Pipeline Job
```bash
# 1. 新建任务 -> Pipeline
# 2. Pipeline配置选择"Pipeline script from SCM"
# 3. 配置Git仓库（包含Jenkinsfile）
# 4. Jenkins自动读取Jenkinsfile执行
```

## 5. Jenkins 插件系统详解

### 🔌 插件的作用
Jenkins插件就像手机APP，扩展Jenkins的功能：

```bash
# 核心Jenkins = 手机系统
# 插件 = 各种APP应用
# 通过插件可以：
# - 集成Git/SVN等版本控制
# - 连接Docker/Kubernetes
# - 发送邮件/钉钉通知
# - 代码质量检查
# - 自动化测试
```

### 📦 常用插件分类

#### 版本控制插件
```bash
Git Plugin          # Git仓库集成
GitHub Plugin       # GitHub集成
GitLab Plugin       # GitLab集成
```

#### 构建工具插件
```bash
Maven Integration   # Maven项目构建
Gradle Plugin       # Gradle项目构建
NodeJS Plugin       # Node.js项目构建
```

#### 部署插件
```bash
Docker Plugin       # Docker容器部署
Kubernetes Plugin   # K8s集成
SSH Plugin          # SSH远程部署
```

#### 通知插件
```bash
Email Extension     # 邮件通知
DingTalk Plugin     # 钉钉通知
Slack Plugin        # Slack通知
```

### 🛠️ 插件安装和使用

#### 安装插件
```bash
# 1. 系统管理 -> 插件管理
# 2. 可选插件 -> 搜索插件名称
# 3. 勾选插件 -> 点击安装
# 4. 重启Jenkins生效
```

#### 使用插件示例：Git Plugin
```groovy
pipeline {
    agent any
    stages {
        stage('代码检出') {
            steps {
                // Git插件提供的功能
                git branch: 'main', 
                    url: 'https://github.com/user/repo.git'
            }
        }
    }
}
```

## 6. 自动触发构建配置

### 🎯 触发方式对比

#### 1. 手动触发
```bash
# 最简单的方式，点击"立即构建"按钮
# 适用场景：测试、紧急发布
```

#### 2. 定时触发
```bash
# 在Job配置中设置"构建触发器"
# 使用Cron表达式：
H 2 * * *     # 每天凌晨2点构建
H/15 * * * *  # 每15分钟构建一次
```

#### 3. 代码提交触发（推荐）
```bash
# 配置Webhook，代码push时自动构建
# GitLab配置：
# 项目设置 -> Webhooks -> 添加Webhook
# URL: http://jenkins-server:8080/project/your-job
```

#### 4. Pipeline中的触发配置
```groovy
pipeline {
    agent any
    
    triggers {
        // 定时触发
        cron('H 2 * * *')
        
        // 轮询SCM变化
        pollSCM('H/5 * * * *')
    }
    
    stages {
        // 构建阶段...
    }
}
```

## 7. 实战演示：完整的CI/CD流程

### 🎬 Demo场景
创建一个简单的网站项目，实现从代码提交到自动部署的完整流程。

#### 步骤1：准备代码仓库
```bash
# 在Gitea中创建新仓库
# 1. 访问 http://localhost:3000
# 2. 注册用户并登录
# 3. 创建新仓库：demo-website
```

#### 步骤2：上传项目代码
```bash
# 创建简单的网站项目
mkdir demo-website
cd demo-website

# 创建网站文件
echo '<!DOCTYPE html>
<html>
<head><title>CI/CD Demo</title></head>
<body>
    <h1>Hello Jenkins CI/CD!</h1>
    <p>这是通过Jenkins自动部署的网站</p>
    <p>构建时间：$(date)</p>
</body>
</html>' > index.html

# 创建Jenkinsfile
echo 'pipeline {
    agent any
    stages {
        stage("构建") {
            steps {
                echo "开始构建网站..."
                sh "ls -la"
            }
        }
        stage("部署") {
            steps {
                echo "部署到Nginx服务器..."
                sh "cp index.html /var/www/html/"
            }
        }
    }
}' > Jenkinsfile

# 提交到Git仓库
git init
git add .
git commit -m "初始化CI/CD演示项目"
git remote add origin http://localhost:3000/your-username/demo-website.git
git push -u origin main
```

#### 步骤3：在Jenkins中创建Pipeline项目
```bash
# 1. 访问 http://localhost:8080
# 2. 新建任务 -> Pipeline
# 3. 任务名称：demo-website-pipeline
# 4. Pipeline配置：
#    - Definition: Pipeline script from SCM
#    - SCM: Git
#    - Repository URL: http://gitea:3000/your-username/demo-website.git
#    - Branch: */main
```

#### 步骤4：配置自动触发
```bash
# 在Gitea中配置Webhook：
# 仓库设置 -> Webhooks -> 添加Webhook
# URL: http://jenkins:8080/project/demo-website-pipeline
# 触发事件：Push events
```

#### 步骤5：测试完整流程
```bash
# 修改index.html文件
echo '<!DOCTYPE html>
<html>
<head><title>CI/CD Demo v2</title></head>
<body>
    <h1>Hello Jenkins CI/CD v2!</h1>
    <p>这是更新后的版本</p>
</body>
</html>' > index.html

# 提交更改
git add .
git commit -m "更新网站内容"
git push

# Jenkins会自动检测到代码变化并开始构建
# 构建完成后，访问 http://localhost:8082 查看部署结果
```

## 8. 常见问题和解决方案

### ❌ 问题1：Jenkins无法连接到Git仓库
```bash
# 解决方案：
# 1. 检查网络连通性
docker exec jenkins-master ping gitea

# 2. 配置Git凭据
# Jenkins管理 -> 凭据管理 -> 添加用户名密码凭据

# 3. 在Pipeline中使用凭据
git branch: 'main', 
    credentialsId: 'gitea-credentials',
    url: 'http://gitea:3000/user/repo.git'
```

### ❌ 问题2：构建失败，权限不足
```bash
# 解决方案：
# 1. 检查Jenkins用户权限
docker exec jenkins-master whoami

# 2. 修改文件权限
docker exec jenkins-master chmod +x /var/jenkins_home/workspace/your-job/script.sh

# 3. 使用sudo（不推荐）
# 在Jenkinsfile中：sh 'sudo your-command'
```

### ❌ 问题3：Docker in Docker问题
```bash
# 解决方案：
# 1. 确保docker.sock已挂载
volumes:
  - /var/run/docker.sock:/var/run/docker.sock

# 2. 安装Docker客户端
# 在Jenkinsfile中：
sh 'curl -fsSL https://get.docker.com | sh'

# 3. 使用Docker Agent
agent {
    docker {
        image 'docker:latest'
    }
}
```

## 9. 调试和监控技巧

### 🔍 调试Pipeline
```groovy
pipeline {
    agent any
    stages {
        stage('调试信息') {
            steps {
                // 打印环境变量
                sh 'env | sort'
                
                // 打印工作目录内容
                sh 'ls -la'
                
                // 打印系统信息
                sh 'uname -a'
                
                // 自定义调试信息
                echo "当前分支: ${env.BRANCH_NAME}"
                echo "构建号: ${env.BUILD_NUMBER}"
            }
        }
    }
}
```

### 📊 监控构建状态
```bash
# 1. 构建历史查看
# 项目页面 -> 构建历史 -> 点击具体构建号

# 2. 控制台输出
# 构建详情页面 -> Console Output

# 3. Pipeline步骤视图
# 构建详情页面 -> Pipeline Steps

# 4. 构建趋势图
# 项目主页显示构建成功率趋势
```

这个完整指南涵盖了Jenkins的核心概念和实际操作，帮助你从零开始掌握CI/CD的实践技能！