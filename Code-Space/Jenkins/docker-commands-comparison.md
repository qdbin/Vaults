# Docker Run vs Docker Compose 详细对比

## 1. 启动方式对比

### Docker Run 方式（传统方式）
```bash
# 需要逐个启动，顺序很重要
docker run -d --name jenkins-master -p 8080:8080 jenkins/jenkins:lts
docker run -d --name gitea -p 3000:3000 gitea/gitea:latest  
docker run -d --name nginx -p 8082:80 nginx:alpine

# 创建网络让容器互相通信
docker network create ci-cd-network
docker network connect ci-cd-network jenkins-master
docker network connect ci-cd-network gitea
docker network connect ci-cd-network nginx
```

### Docker Compose 方式（现代方式）
```bash
# 一条命令搞定所有容器
docker-compose up -d

# 相当于自动执行了上面所有的docker run命令
```

## 2. 管理操作对比

| 操作 | Docker Run | Docker Compose |
|------|------------|----------------|
| 启动所有服务 | `docker start jenkins gitea nginx` | `docker-compose up -d` |
| 停止所有服务 | `docker stop jenkins gitea nginx` | `docker-compose down` |
| 查看日志 | `docker logs jenkins` | `docker-compose logs jenkins` |
| 重启服务 | `docker restart jenkins` | `docker-compose restart jenkins` |
| 删除所有 | `docker rm -f jenkins gitea nginx` | `docker-compose down -v` |

## 3. 配置管理对比

### Docker Run 配置（分散管理）
```bash
# 参数分散在各个命令中，难以管理
docker run -d \
  --name jenkins-master \
  -p 8080:8080 \
  -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e JAVA_OPTS=-Djenkins.install.runSetupWizard=false \
  --restart unless-stopped \
  jenkins/jenkins:lts
```

### Docker Compose 配置（集中管理）
```yaml
# 所有配置集中在一个文件中，清晰明了
services:
  jenkins:
    image: jenkins/jenkins:lts
    container_name: jenkins-master
    ports:
      - "8080:8080"
      - "50000:50000"
    volumes:
      - jenkins_home:/var/jenkins_home
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - JAVA_OPTS=-Djenkins.install.runSetupWizard=false
    restart: unless-stopped
```

## 4. 容器隔离性说明

### 重要概念澄清
- **每个service仍然是独立的容器**
- **不是共享同一个系统内核的进程**
- **仍然有完整的文件系统隔离**

### 网络通信示例
```bash
# 在jenkins容器内可以通过服务名访问其他容器
docker exec jenkins-master ping gitea        # ✅ 可以ping通
docker exec jenkins-master ping nginx        # ✅ 可以ping通

# 但文件系统仍然完全隔离
docker exec jenkins-master ls /              # 只能看到jenkins容器的文件
docker exec gitea ls /                       # 只能看到gitea容器的文件
```

## 5. 实际使用建议

### 什么时候用Docker Run
- 单个容器测试
- 临时启动某个服务
- 学习Docker基础概念

### 什么时候用Docker Compose
- 多容器应用（推荐）
- 开发环境搭建
- 生产环境部署
- 需要容器间通信的场景

## 6. 企业级最佳实践

### 开发环境
```bash
# 开发时快速启动整套环境
docker-compose up -d
```

### 生产环境
```bash
# 生产环境通常会分离部署
docker-compose -f docker-compose.prod.yml up -d
```

### 扩展性考虑
```yaml
# 可以轻松扩展服务实例
services:
  jenkins:
    deploy:
      replicas: 2  # 启动2个Jenkins实例
```