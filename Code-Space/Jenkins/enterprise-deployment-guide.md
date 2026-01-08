# 企业级 GitLab + Jenkins 分离部署完整指南

## 1. 为什么要分离部署？深度解析

### 🤔 问题本质
你的疑问很有道理！让我用大白话解释为什么企业要分离部署：

### 🏭 企业级架构类比
```
传统工厂模式：
原料仓库 (GitLab) ←→ 生产车间 (Jenkins) ←→ 成品仓库 (部署服务器)
     ↓                    ↓                      ↓
   存储代码              执行构建                运行应用
```

### 💡 分离部署的核心原因

#### 1. **职责分离原则**
```bash
GitLab 职责：
├── 代码版本管理
├── 代码审查 (Merge Request)
├── 问题跟踪 (Issues)
├── Wiki文档管理
└── 用户权限管理

Jenkins 职责：
├── 构建任务调度
├── 自动化测试执行
├── 部署流程管理
├── 构建历史记录
└── 插件生态集成
```

#### 2. **资源隔离和性能优化**
```bash
GitLab 服务器配置：
├── 高存储容量 (代码仓库、文件存储)
├── 高内存 (Git操作、数据库)
├── 网络带宽 (代码克隆、推送)
└── 数据库性能 (PostgreSQL)

Jenkins 服务器配置：
├── 高CPU性能 (编译构建)
├── 大内存 (并发构建)
├── 快速磁盘 (临时文件、缓存)
└── 网络连接 (下载依赖、部署)
```

#### 3. **安全性考虑**
```bash
安全边界划分：
GitLab (代码安全)
├── 代码访问权限控制
├── 分支保护策略
├── 审计日志
└── 备份策略

Jenkins (构建安全)
├── 构建环境隔离
├── 凭据管理
├── 插件安全
└── 部署权限控制
```

## 2. 企业级分离部署架构设计

### 🏗️ 标准三层架构
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   GitLab 服务器   │    │  Jenkins 服务器  │    │   部署目标服务器   │
│                 │    │                 │    │                 │
│ ├── Git 仓库     │    │ ├── Master 节点  │    │ ├── 生产环境     │
│ ├── 用户管理     │◄──►│ ├── Agent 节点   │◄──►│ ├── 测试环境     │
│ ├── CI/CD 触发   │    │ ├── 构建队列     │    │ ├── 预发布环境   │
│ └── Webhook     │    │ └── 部署脚本     │    │ └── 监控日志     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
       ↑                        ↑                        ↑
   开发人员提交代码          自动化构建流程           应用运行环境
```

### 🔧 实际部署配置示例

#### GitLab 服务器配置 (gitlab-server.yml)
```yaml
version: '3.8'
services:
  gitlab:
    image: gitlab/gitlab-ce:latest
    hostname: 'gitlab.company.com'
    ports:
      - '80:80'
      - '443:443'
      - '22:22'
    volumes:
      - gitlab_config:/etc/gitlab
      - gitlab_logs:/var/log/gitlab
      - gitlab_data:/var/opt/gitlab
    environment:
      GITLAB_OMNIBUS_CONFIG: |
        external_url 'https://gitlab.company.com'
        # 邮件配置
        gitlab_rails['smtp_enable'] = true
        gitlab_rails['smtp_address'] = "smtp.company.com"
        # 备份配置
        gitlab_rails['backup_keep_time'] = 604800
        # 性能优化
        postgresql['shared_buffers'] = "256MB"
        postgresql['max_connections'] = 200
    restart: unless-stopped

volumes:
  gitlab_config:
  gitlab_logs:
  gitlab_data:
```

#### Jenkins 服务器配置 (jenkins-server.yml)
```yaml
version: '3.8'
services:
  jenkins-master:
    image: jenkins/jenkins:lts
    ports:
      - '8080:8080'
      - '50000:50000'
    volumes:
      - jenkins_home:/var/jenkins_home
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - JAVA_OPTS=-Xmx2g -Xms1g
    restart: unless-stopped

  jenkins-agent-1:
    image: jenkins/ssh-agent:latest
    environment:
      - JENKINS_AGENT_SSH_PUBKEY=ssh-rsa AAAAB3NzaC1yc2E...
    volumes:
      - agent1_workspace:/home/jenkins
    restart: unless-stopped

  jenkins-agent-2:
    image: jenkins/ssh-agent:latest
    environment:
      - JENKINS_AGENT_SSH_PUBKEY=ssh-rsa AAAAB3NzaC1yc2E...
    volumes:
      - agent2_workspace:/home/jenkins
    restart: unless-stopped

volumes:
  jenkins_home:
  agent1_workspace:
  agent2_workspace:
```

## 3. GitLab 环境配置详解

### 🛠️ GitLab 需要的环境和工具

#### 基础运行环境
```bash
# GitLab 内置环境（无需额外安装）
├── Ruby on Rails (Web框架)
├── PostgreSQL (数据库)
├── Redis (缓存)
├── Nginx (Web服务器)
├── Git (版本控制)
└── Sidekiq (后台任务处理)
```

#### GitLab CI Runner 环境配置
```yaml
# .gitlab-ci.yml 示例
stages:
  - build
  - test
  - deploy

variables:
  MAVEN_OPTS: "-Dmaven.repo.local=$CI_PROJECT_DIR/.m2/repository"

# 构建阶段 - 需要Java和Maven环境
build_job:
  stage: build
  image: maven:3.8.4-openjdk-11  # 指定构建环境镜像
  script:
    - mvn clean compile
  artifacts:
    paths:
      - target/
  cache:
    paths:
      - .m2/repository/

# 测试阶段 - 需要测试工具
test_job:
  stage: test
  image: maven:3.8.4-openjdk-11
  script:
    - mvn test
  artifacts:
    reports:
      junit: target/surefire-reports/TEST-*.xml

# 部署阶段 - 需要部署工具
deploy_job:
  stage: deploy
  image: alpine:latest
  before_script:
    - apk add --no-cache curl
    - apk add --no-cache openssh-client
  script:
    - scp target/*.jar user@deploy-server:/opt/app/
    - ssh user@deploy-server "systemctl restart myapp"
  only:
    - main
```

### 🔧 GitLab Runner 配置
```bash
# 1. 安装GitLab Runner
curl -L https://packages.gitlab.com/install/repositories/runner/gitlab-runner/script.rpm.sh | sudo bash
sudo yum install gitlab-runner

# 2. 注册Runner
sudo gitlab-runner register \
  --url "https://gitlab.company.com/" \
  --registration-token "your-token" \
  --description "docker-runner" \
  --executor "docker" \
  --docker-image "alpine:latest"

# 3. 配置Runner环境
# /etc/gitlab-runner/config.toml
[[runners]]
  name = "docker-runner"
  url = "https://gitlab.company.com/"
  token = "your-token"
  executor = "docker"
  [runners.docker]
    image = "alpine:latest"
    privileged = true
    volumes = ["/cache", "/var/run/docker.sock:/var/run/docker.sock"]
```

## 4. Jenkins 环境配置详解

### 🎯 Jenkins 服务器环境需求

#### Jenkins Master 环境
```bash
# Jenkins Master 只需要：
├── Java 运行环境 (JDK 11+)
├── Jenkins WAR 包
├── 插件管理
└── 配置管理

# 不需要具体的构建工具！
# 构建工具安装在Agent上
```

#### Jenkins Agent 环境配置
```bash
# Agent-1: Java项目构建环境
FROM openjdk:11-jdk
RUN apt-get update && apt-get install -y \
    maven \
    gradle \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Agent-2: Node.js项目构建环境  
FROM node:16-alpine
RUN apk add --no-cache \
    git \
    python3 \
    make \
    g++

# Agent-3: Docker构建环境
FROM docker:latest
RUN apk add --no-cache \
    git \
    curl \
    bash
```

### 🔧 Jenkins 插件配置示例
```groovy
// Jenkinsfile - 使用不同Agent构建不同类型项目
pipeline {
    agent none
    
    stages {
        stage('Java构建') {
            agent {
                label 'java-agent'  // 指定Java环境的Agent
            }
            steps {
                sh 'mvn clean package'
            }
        }
        
        stage('前端构建') {
            agent {
                label 'nodejs-agent'  // 指定Node.js环境的Agent
            }
            steps {
                sh 'npm install'
                sh 'npm run build'
            }
        }
        
        stage('Docker构建') {
            agent {
                label 'docker-agent'  // 指定Docker环境的Agent
            }
            steps {
                sh 'docker build -t myapp:${BUILD_NUMBER} .'
                sh 'docker push registry.company.com/myapp:${BUILD_NUMBER}'
            }
        }
    }
}
```

## 5. 企业级完整Demo示例

### 🎬 场景：Java Web应用的完整CI/CD流程

#### 项目结构
```
demo-java-web/
├── src/
│   ├── main/java/com/company/app/
│   └── test/java/com/company/app/
├── pom.xml
├── Dockerfile
├── .gitlab-ci.yml
├── Jenkinsfile
└── deploy/
    ├── docker-compose.yml
    └── nginx.conf
```

#### 步骤1：GitLab仓库配置
```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - package
  - trigger-jenkins

variables:
  MAVEN_OPTS: "-Dmaven.repo.local=$CI_PROJECT_DIR/.m2/repository"

maven-build:
  stage: build
  image: maven:3.8.4-openjdk-11
  script:
    - mvn clean compile
  artifacts:
    paths:
      - target/classes/
  cache:
    paths:
      - .m2/repository/

maven-test:
  stage: test
  image: maven:3.8.4-openjdk-11
  script:
    - mvn test
  artifacts:
    reports:
      junit: target/surefire-reports/TEST-*.xml

maven-package:
  stage: package
  image: maven:3.8.4-openjdk-11
  script:
    - mvn package -DskipTests
  artifacts:
    paths:
      - target/*.jar
  only:
    - main

trigger-jenkins:
  stage: trigger-jenkins
  image: alpine:latest
  before_script:
    - apk add --no-cache curl
  script:
    - |
      curl -X POST \
        "http://jenkins.company.com:8080/job/demo-java-web-deploy/buildWithParameters" \
        --user "jenkins-user:jenkins-token" \
        --data "GITLAB_COMMIT_SHA=$CI_COMMIT_SHA" \
        --data "GITLAB_BRANCH=$CI_COMMIT_REF_NAME"
  only:
    - main
```

#### 步骤2：Jenkins Pipeline配置
```groovy
// Jenkinsfile
pipeline {
    agent none
    
    parameters {
        string(name: 'GITLAB_COMMIT_SHA', defaultValue: '', description: 'GitLab提交SHA')
        string(name: 'GITLAB_BRANCH', defaultValue: 'main', description: 'GitLab分支')
    }
    
    environment {
        DOCKER_REGISTRY = 'registry.company.com'
        APP_NAME = 'demo-java-web'
        IMAGE_TAG = "${BUILD_NUMBER}-${params.GITLAB_COMMIT_SHA?.take(8)}"
    }
    
    stages {
        stage('从GitLab获取制品') {
            agent {
                label 'docker-agent'
            }
            steps {
                script {
                    // 从GitLab下载构建制品
                    sh """
                        curl -H "PRIVATE-TOKEN: \${GITLAB_TOKEN}" \
                             -o app.jar \
                             "https://gitlab.company.com/api/v4/projects/123/jobs/artifacts/${params.GITLAB_BRANCH}/raw/target/app.jar?job=maven-package"
                    """
                }
            }
        }
        
        stage('构建Docker镜像') {
            agent {
                label 'docker-agent'
            }
            steps {
                script {
                    sh """
                        docker build -t ${DOCKER_REGISTRY}/${APP_NAME}:${IMAGE_TAG} .
                        docker push ${DOCKER_REGISTRY}/${APP_NAME}:${IMAGE_TAG}
                    """
                }
            }
        }
        
        stage('部署到测试环境') {
            agent {
                label 'deploy-agent'
            }
            steps {
                script {
                    sh """
                        # 更新测试环境
                        ssh test-server "docker pull ${DOCKER_REGISTRY}/${APP_NAME}:${IMAGE_TAG}"
                        ssh test-server "docker stop ${APP_NAME}-test || true"
                        ssh test-server "docker run -d --name ${APP_NAME}-test -p 8080:8080 ${DOCKER_REGISTRY}/${APP_NAME}:${IMAGE_TAG}"
                    """
                }
            }
        }
        
        stage('自动化测试') {
            agent {
                label 'test-agent'
            }
            steps {
                script {
                    sh """
                        # 等待应用启动
                        sleep 30
                        
                        # 执行集成测试
                        curl -f http://test-server:8080/health || exit 1
                        
                        # 执行API测试
                        newman run api-tests.json --environment test-env.json
                    """
                }
            }
        }
        
        stage('部署到生产环境') {
            agent {
                label 'deploy-agent'
            }
            when {
                branch 'main'
            }
            input {
                message "是否部署到生产环境？"
                ok "部署"
                parameters {
                    choice(name: 'DEPLOY_STRATEGY', choices: ['蓝绿部署', '滚动更新'], description: '部署策略')
                }
            }
            steps {
                script {
                    if (params.DEPLOY_STRATEGY == '蓝绿部署') {
                        sh """
                            # 蓝绿部署逻辑
                            ssh prod-server "docker-compose -f docker-compose.blue-green.yml up -d --scale app=2"
                            sleep 60
                            ssh prod-server "nginx -s reload"  # 切换流量
                        """
                    } else {
                        sh """
                            # 滚动更新逻辑
                            ssh prod-server "docker service update --image ${DOCKER_REGISTRY}/${APP_NAME}:${IMAGE_TAG} prod_app"
                        """
                    }
                }
            }
        }
    }
    
    post {
        success {
            // 发送成功通知
            dingtalk (
                robot: 'jenkins-bot',
                message: "✅ ${APP_NAME} 部署成功！\n版本：${IMAGE_TAG}\n分支：${params.GITLAB_BRANCH}"
            )
        }
        failure {
            // 发送失败通知
            dingtalk (
                robot: 'jenkins-bot',
                message: "❌ ${APP_NAME} 部署失败！\n请检查构建日志：${BUILD_URL}"
            )
        }
    }
}
```

#### 步骤3：部署服务器配置
```yaml
# deploy/docker-compose.yml
version: '3.8'
services:
  app:
    image: registry.company.com/demo-java-web:latest
    ports:
      - "8080:8080"
    environment:
      - SPRING_PROFILES_ACTIVE=production
      - DATABASE_URL=jdbc:mysql://db:3306/appdb
    depends_on:
      - db
      - redis
    restart: unless-stopped

  db:
    image: mysql:8.0
    environment:
      - MYSQL_ROOT_PASSWORD=rootpass
      - MYSQL_DATABASE=appdb
    volumes:
      - mysql_data:/var/lib/mysql
    restart: unless-stopped

  redis:
    image: redis:alpine
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  mysql_data:
```

## 6. 分离部署的优势总结

### ✅ 技术优势
```bash
1. 专业化分工
   ├── GitLab专注代码管理和协作
   ├── Jenkins专注构建和部署
   └── 各自发挥最大优势

2. 性能优化
   ├── 资源配置针对性优化
   ├── 负载分散，避免单点瓶颈
   └── 可独立扩展

3. 安全性提升
   ├── 权限边界清晰
   ├── 攻击面分散
   └── 审计追踪完整
```

### ✅ 运维优势
```bash
1. 独立维护
   ├── 可独立升级更新
   ├── 故障影响范围小
   └── 备份恢复策略独立

2. 团队协作
   ├── 开发团队管理GitLab
   ├── 运维团队管理Jenkins
   └── 职责分工明确
```

### ✅ 成本优势
```bash
1. 资源利用率
   ├── 按需配置硬件资源
   ├── 避免资源浪费
   └── 成本控制精确

2. 扩展性
   ├── 可独立水平扩展
   ├── 支持多环境部署
   └── 适应业务增长
```

这就是为什么企业级部署要分离GitLab和Jenkins的完整原因和实践方案！每个组件都有自己的专业领域，分离部署能够最大化发挥各自的优势。