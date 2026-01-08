# 改进版 Jenkins + GitLab CI/CD 环境

基于 [万字干货! 使用docker部署jenkins和gitlab](https://www.cnblogs.com/wbourne/p/17035591.html) 的方案改进。

## 主要改进

### 1. 架构优化
- **移除冗余服务**: 去掉了Gitea，专注于GitLab + Jenkins的完整CI/CD流程
- **添加应用容器**: 新增Java应用部署容器，实现真正的自动化部署
- **网络优化**: 使用自定义子网，便于容器间通信

### 2. 配置改进
- **绑定挂载**: 使用绑定挂载替代命名卷，便于数据管理和备份
- **内存优化**: 针对GitLab进行内存使用优化
- **自动化配置**: 应用容器自动安装SSH和Java环境

### 3. 实际可用性
- **完整CI/CD**: 支持代码推送→自动构建→自动部署的完整流程
- **生产级配置**: SSH配置、自动化脚本、日志管理
- **易于扩展**: 可以轻松添加更多应用容器

## 服务说明

| 服务 | 端口 | 用途 | 访问地址 |
|------|------|------|----------|
| Jenkins | 8080 | CI/CD平台 | http://localhost:8080 |
| GitLab | 8081 | Git仓库 | http://localhost:8081 |
| Java应用服务器 | 31808 | 应用部署 | http://localhost:31808 |
| Web服务器 | 8082 | 前端部署 | http://localhost:8082 |

## 快速开始

### 1. 启动服务
```bash
docker-compose up -d
```

### 2. 等待服务启动
- GitLab需要3-5分钟启动
- Jenkins需要1-2分钟启动

### 3. 获取初始密码

**GitLab密码**:
```bash
docker exec -it gitlab-ce grep 'Password:' /etc/gitlab/initial_root_password
```

**Jenkins密码**:
```bash
docker exec -it jenkins-master cat /var/jenkins_home/secrets/initialAdminPassword
```

### 4. 配置Jenkins

1. 访问 http://localhost:8080
2. 输入初始密码
3. 安装推荐插件
4. 额外安装插件:
   - Publish Over SSH
   - Maven Integration  
   - Build Authorization Token Root

### 5. 配置SSH连接

在Jenkins中配置SSH连接到应用服务器:
- 主机: `app-server`
- 用户名: `root`
- 密码: `password123`
- 远程目录: `/root`

## 完整CI/CD流程

### 1. 创建GitLab项目
1. 访问 http://localhost:8081
2. 使用root账户登录
3. 创建新项目

### 2. 配置Jenkins任务
1. 创建Maven项目
2. 配置Git仓库地址: `http://gitlab/root/your-project.git`
3. 配置构建后操作:
   - 传输JAR包到应用服务器
   - 执行启动脚本: `bash /root/scripts/start.sh`

### 3. 配置自动触发
1. 在Jenkins中设置构建触发器
2. 在GitLab中配置Webhook
3. 实现代码推送自动部署

## 目录结构

```
Jenkins/
├── docker-compose.yml          # 主配置文件
├── scripts/
│   └── start.sh               # 应用启动脚本
├── jenkins-data/              # Jenkins数据目录
├── gitlab/                    # GitLab数据目录
│   ├── config/
│   ├── logs/
│   └── data/
├── app/                       # 应用部署目录
└── html/                      # Web服务器内容
```

## 故障排除

### 1. 内存不足
如果系统内存小于16GB，可以:
- 减少GitLab的worker进程数
- 使用Gitea替代GitLab
- 调整PostgreSQL配置

### 2. 端口冲突
如果端口被占用，修改docker-compose.yml中的端口映射

### 3. 容器启动失败
```bash
# 查看日志
docker-compose logs [service-name]

# 重启服务
docker-compose restart [service-name]
```

## 与原方案对比

| 方面 | 原docker-compose.yml | 改进版 | 网站方案 |
|------|---------------------|--------|----------|
| 服务数量 | 4个(Jenkins+GitLab+Gitea+Nginx) | 4个(Jenkins+GitLab+App+Web) | 3个(Jenkins+GitLab+App) |
| 部署目标 | 仅Nginx演示 | Java应用+Web应用 | Java应用 |
| 网络配置 | 默认桥接 | 自定义子网 | 自定义子网 |
| 数据管理 | 命名卷 | 绑定挂载 | 绑定挂载 |
| CI/CD完整性 | 不完整 | 完整 | 完整 |
| 生产可用性 | 学习环境 | 接近生产 | 生产级 |

这个改进版本结合了原docker-compose.yml的易用性和网站方案的完整性，更适合学习和实践CI/CD流程。