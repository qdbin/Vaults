// 🏢 企业级Jenkins Pipeline完整示例
// 适用于Spring Boot微服务项目

pipeline {
    agent none
    
    // 🔧 全局工具配置
    tools {
        maven 'Maven-3.8.6'
        jdk 'OpenJDK-11'
        nodejs 'NodeJS-16'
    }
    
    // 🌍 环境变量配置
    environment {
        // 应用配置
        APP_NAME = 'user-service'
        APP_VERSION = "${env.BUILD_NUMBER}"
        
        // Docker配置
        DOCKER_REGISTRY = 'harbor.company.com'
        DOCKER_NAMESPACE = 'microservices'
        DOCKER_IMAGE = "${DOCKER_REGISTRY}/${DOCKER_NAMESPACE}/${APP_NAME}"
        
        // Kubernetes配置
        K8S_NAMESPACE_DEV = 'development'
        K8S_NAMESPACE_TEST = 'testing'
        K8S_NAMESPACE_PROD = 'production'
        
        // 质量门禁配置
        SONAR_PROJECT_KEY = "${APP_NAME}"
        COVERAGE_THRESHOLD = '80'
        
        // 通知配置
        SLACK_CHANNEL = '#ci-cd-notifications'
        EMAIL_RECIPIENTS = 'dev-team@company.com'
    }
    
    // ⚙️ 参数化构建
    parameters {
        choice(
            name: 'DEPLOY_ENV',
            choices: ['dev', 'test', 'prod'],
            description: '选择部署环境'
        )
        choice(
            name: 'DEPLOY_STRATEGY',
            choices: ['rolling', 'blue-green', 'canary'],
            description: '选择部署策略'
        )
        booleanParam(
            name: 'SKIP_TESTS',
            defaultValue: false,
            description: '跳过测试阶段'
        )
        booleanParam(
            name: 'FORCE_DEPLOY',
            defaultValue: false,
            description: '强制部署（跳过质量门禁）'
        )
        string(
            name: 'CUSTOM_TAG',
            defaultValue: '',
            description: '自定义镜像标签（可选）'
        )
    }
    
    // 🔄 触发器配置
    triggers {
        // 主分支每天凌晨2点自动构建
        cron(env.BRANCH_NAME == 'main' ? 'H 2 * * *' : '')
        // 开发分支代码变更时触发
        pollSCM(env.BRANCH_NAME == 'develop' ? 'H/5 * * * *' : '')
    }
    
    // 📋 构建阶段
    stages {
        
        // 🔍 环境准备和检查
        stage('环境准备') {
            agent {
                label 'linux && docker'
            }
            steps {
                script {
                    // 设置构建显示名称
                    currentBuild.displayName = "#${env.BUILD_NUMBER}-${params.DEPLOY_ENV}"
                    currentBuild.description = "分支: ${env.BRANCH_NAME}, 环境: ${params.DEPLOY_ENV}"
                    
                    // 环境检查
                    sh '''
                        echo "=== 环境信息检查 ==="
                        echo "Java版本: $(java -version 2>&1 | head -1)"
                        echo "Maven版本: $(mvn -version | head -1)"
                        echo "Docker版本: $(docker --version)"
                        echo "Kubectl版本: $(kubectl version --client --short)"
                        echo "构建时间: $(date)"
                        echo "构建节点: ${NODE_NAME}"
                    '''
                }
            }
        }
        
        // 📥 代码检出
        stage('代码检出') {
            agent {
                label 'linux'
            }
            steps {
                checkout scm
                script {
                    // 获取Git信息
                    env.GIT_COMMIT_SHORT = sh(
                        script: 'git rev-parse --short HEAD',
                        returnStdout: true
                    ).trim()
                    env.GIT_AUTHOR = sh(
                        script: 'git log -1 --pretty=format:"%an"',
                        returnStdout: true
                    ).trim()
                    env.GIT_MESSAGE = sh(
                        script: 'git log -1 --pretty=format:"%s"',
                        returnStdout: true
                    ).trim()
                    
                    echo "提交信息: ${env.GIT_MESSAGE}"
                    echo "提交作者: ${env.GIT_AUTHOR}"
                    echo "提交哈希: ${env.GIT_COMMIT_SHORT}"
                }
            }
        }
        
        // 🔧 依赖管理
        stage('依赖安装') {
            agent {
                docker {
                    image 'maven:3.8.6-openjdk-11'
                    args '-v /root/.m2:/root/.m2'
                }
            }
            steps {
                sh '''
                    echo "=== Maven依赖下载 ==="
                    mvn dependency:resolve
                    mvn dependency:resolve-sources
                '''
            }
        }
        
        // 🏗️ 编译构建
        stage('编译构建') {
            agent {
                docker {
                    image 'maven:3.8.6-openjdk-11'
                    args '-v /root/.m2:/root/.m2'
                }
            }
            steps {
                sh '''
                    echo "=== 编译Java代码 ==="
                    mvn clean compile
                '''
            }
        }
        
        // 🧪 测试阶段
        stage('测试执行') {
            when {
                not { params.SKIP_TESTS }
            }
            parallel {
                // 单元测试
                stage('单元测试') {
                    agent {
                        docker {
                            image 'maven:3.8.6-openjdk-11'
                            args '-v /root/.m2:/root/.m2'
                        }
                    }
                    steps {
                        sh '''
                            echo "=== 执行单元测试 ==="
                            mvn test
                        '''
                    }
                    post {
                        always {
                            // 发布测试结果
                            junit 'target/surefire-reports/*.xml'
                            // 发布覆盖率报告
                            publishHTML([
                                allowMissing: false,
                                alwaysLinkToLastBuild: true,
                                keepAll: true,
                                reportDir: 'target/site/jacoco',
                                reportFiles: 'index.html',
                                reportName: '代码覆盖率报告'
                            ])
                        }
                    }
                }
                
                // 集成测试
                stage('集成测试') {
                    agent {
                        docker {
                            image 'maven:3.8.6-openjdk-11'
                            args '-v /root/.m2:/root/.m2 -v /var/run/docker.sock:/var/run/docker.sock'
                        }
                    }
                    steps {
                        sh '''
                            echo "=== 执行集成测试 ==="
                            mvn integration-test
                        '''
                    }
                }
                
                // 前端测试（如果存在）
                stage('前端测试') {
                    when {
                        expression {
                            return fileExists('package.json')
                        }
                    }
                    agent {
                        docker {
                            image 'node:16-alpine'
                        }
                    }
                    steps {
                        sh '''
                            echo "=== 前端测试 ==="
                            npm ci
                            npm run test:ci
                        '''
                    }
                }
            }
        }
        
        // 🔍 代码质量检查
        stage('代码质量分析') {
            agent {
                docker {
                    image 'maven:3.8.6-openjdk-11'
                    args '-v /root/.m2:/root/.m2'
                }
            }
            steps {
                script {
                    withSonarQubeEnv('SonarQube') {
                        sh '''
                            echo "=== SonarQube代码扫描 ==="
                            mvn sonar:sonar \
                                -Dsonar.projectKey=${SONAR_PROJECT_KEY} \
                                -Dsonar.projectName=${APP_NAME} \
                                -Dsonar.projectVersion=${APP_VERSION}
                        '''
                    }
                }
            }
        }
        
        // 🚪 质量门禁
        stage('质量门禁') {
            when {
                not { params.FORCE_DEPLOY }
            }
            steps {
                script {
                    timeout(time: 5, unit: 'MINUTES') {
                        def qg = waitForQualityGate()
                        if (qg.status != 'OK') {
                            error "质量门禁失败: ${qg.status}"
                        }
                    }
                }
            }
        }
        
        // 📦 应用打包
        stage('应用打包') {
            agent {
                docker {
                    image 'maven:3.8.6-openjdk-11'
                    args '-v /root/.m2:/root/.m2'
                }
            }
            steps {
                sh '''
                    echo "=== Maven打包 ==="
                    mvn package -DskipTests
                '''
                
                // 归档构建产物
                archiveArtifacts artifacts: 'target/*.jar', fingerprint: true
                
                // 保存构建信息
                script {
                    writeFile file: 'build-info.json', text: """
                    {
                        "appName": "${env.APP_NAME}",
                        "version": "${env.APP_VERSION}",
                        "gitCommit": "${env.GIT_COMMIT_SHORT}",
                        "gitAuthor": "${env.GIT_AUTHOR}",
                        "gitMessage": "${env.GIT_MESSAGE}",
                        "buildTime": "${new Date()}",
                        "buildNode": "${env.NODE_NAME}",
                        "buildUrl": "${env.BUILD_URL}"
                    }
                    """
                    archiveArtifacts artifacts: 'build-info.json'
                }
            }
        }
        
        // 🐳 Docker镜像构建
        stage('Docker构建') {
            agent {
                label 'linux && docker'
            }
            steps {
                script {
                    // 确定镜像标签
                    def imageTag = params.CUSTOM_TAG ?: "${env.APP_VERSION}-${env.GIT_COMMIT_SHORT}"
                    env.DOCKER_TAG = imageTag
                    
                    // 构建Docker镜像
                    def image = docker.build("${env.DOCKER_IMAGE}:${imageTag}")
                    
                    // 推送到镜像仓库
                    docker.withRegistry("https://${env.DOCKER_REGISTRY}", 'harbor-credentials') {
                        image.push()
                        image.push('latest')
                    }
                    
                    echo "Docker镜像构建完成: ${env.DOCKER_IMAGE}:${imageTag}"
                }
            }
        }
        
        // 🔒 安全扫描
        stage('安全扫描') {
            parallel {
                // 依赖漏洞扫描
                stage('依赖安全扫描') {
                    agent {
                        docker {
                            image 'maven:3.8.6-openjdk-11'
                            args '-v /root/.m2:/root/.m2'
                        }
                    }
                    steps {
                        sh '''
                            echo "=== OWASP依赖检查 ==="
                            mvn org.owasp:dependency-check-maven:check
                        '''
                    }
                    post {
                        always {
                            publishHTML([
                                allowMissing: true,
                                alwaysLinkToLastBuild: true,
                                keepAll: true,
                                reportDir: 'target',
                                reportFiles: 'dependency-check-report.html',
                                reportName: '依赖安全报告'
                            ])
                        }
                    }
                }
                
                // 镜像安全扫描
                stage('镜像安全扫描') {
                    agent {
                        label 'linux && docker'
                    }
                    steps {
                        script {
                            sh """
                                echo "=== Trivy镜像扫描 ==="
                                docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
                                    aquasec/trivy:latest image \
                                    --format json \
                                    --output trivy-report.json \
                                    ${env.DOCKER_IMAGE}:${env.DOCKER_TAG}
                            """
                        }
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: 'trivy-report.json'
                        }
                    }
                }
            }
        }
        
        // 🚀 部署阶段
        stage('应用部署') {
            agent {
                label 'linux && kubectl'
            }
            steps {
                script {
                    switch(params.DEPLOY_ENV) {
                        case 'dev':
                            deployToKubernetes(env.K8S_NAMESPACE_DEV, params.DEPLOY_STRATEGY)
                            break
                        case 'test':
                            deployToKubernetes(env.K8S_NAMESPACE_TEST, params.DEPLOY_STRATEGY)
                            break
                        case 'prod':
                            // 生产环境需要审批
                            input message: '确认部署到生产环境？', 
                                  ok: '确认部署',
                                  submitterParameter: 'APPROVER'
                            echo "部署审批人: ${env.APPROVER}"
                            deployToKubernetes(env.K8S_NAMESPACE_PROD, params.DEPLOY_STRATEGY)
                            break
                    }
                }
            }
        }
        
        // 🔍 部署验证
        stage('部署验证') {
            agent {
                label 'linux && kubectl'
            }
            steps {
                script {
                    def namespace = getNamespaceByEnv(params.DEPLOY_ENV)
                    
                    // 健康检查
                    timeout(time: 10, unit: 'MINUTES') {
                        sh """
                            echo "=== 等待Pod就绪 ==="
                            kubectl wait --for=condition=ready pod \
                                -l app=${env.APP_NAME} \
                                -n ${namespace} \
                                --timeout=600s
                        """
                    }
                    
                    // 服务可用性检查
                    sh """
                        echo "=== 服务健康检查 ==="
                        kubectl get pods -l app=${env.APP_NAME} -n ${namespace}
                        kubectl get svc -l app=${env.APP_NAME} -n ${namespace}
                    """
                    
                    // API健康检查
                    def serviceUrl = getServiceUrl(params.DEPLOY_ENV)
                    timeout(time: 5, unit: 'MINUTES') {
                        waitUntil {
                            script {
                                def response = sh(
                                    script: "curl -s -o /dev/null -w '%{http_code}' ${serviceUrl}/actuator/health",
                                    returnStdout: true
                                ).trim()
                                return response == '200'
                            }
                        }
                    }
                    
                    echo "✅ 部署验证成功，服务正常运行"
                }
            }
        }
        
        // 📊 性能测试（可选）
        stage('性能测试') {
            when {
                anyOf {
                    expression { params.DEPLOY_ENV == 'test' }
                    expression { params.DEPLOY_ENV == 'prod' }
                }
            }
            agent {
                docker {
                    image 'loadimpact/k6:latest'
                }
            }
            steps {
                script {
                    def serviceUrl = getServiceUrl(params.DEPLOY_ENV)
                    sh """
                        echo "=== K6性能测试 ==="
                        k6 run --vus 10 --duration 30s \
                            -e BASE_URL=${serviceUrl} \
                            performance-tests/load-test.js
                    """
                }
            }
        }
    }
    
    // 📊 构建后处理
    post {
        always {
            script {
                // 清理工作空间
                cleanWs()
                
                // 发送构建通知
                sendNotification(currentBuild.result ?: 'SUCCESS')
            }
        }
        success {
            echo '🎉 构建成功完成！'
        }
        failure {
            echo '❌ 构建失败，请检查日志'
        }
        unstable {
            echo '⚠️ 构建不稳定，存在测试失败'
        }
        aborted {
            echo '🛑 构建被中止'
        }
    }
}

// 🔧 自定义函数定义

/**
 * 部署到Kubernetes集群
 */
def deployToKubernetes(namespace, strategy) {
    echo "开始部署到Kubernetes: ${namespace}, 策略: ${strategy}"
    
    // 准备部署配置
    sh """
        # 替换部署模板中的变量
        envsubst < k8s/deployment-template.yaml > k8s/deployment.yaml
        envsubst < k8s/service-template.yaml > k8s/service.yaml
        
        # 应用配置
        kubectl apply -f k8s/deployment.yaml -n ${namespace}
        kubectl apply -f k8s/service.yaml -n ${namespace}
    """
    
    // 根据策略执行部署
    switch(strategy) {
        case 'rolling':
            rollingUpdate(namespace)
            break
        case 'blue-green':
            blueGreenDeploy(namespace)
            break
        case 'canary':
            canaryDeploy(namespace)
            break
        default:
            rollingUpdate(namespace)
    }
}

/**
 * 滚动更新部署
 */
def rollingUpdate(namespace) {
    sh """
        echo "=== 执行滚动更新 ==="
        kubectl set image deployment/${env.APP_NAME} \
            ${env.APP_NAME}=${env.DOCKER_IMAGE}:${env.DOCKER_TAG} \
            -n ${namespace}
        
        kubectl rollout status deployment/${env.APP_NAME} -n ${namespace}
    """
}

/**
 * 蓝绿部署
 */
def blueGreenDeploy(namespace) {
    sh """
        echo "=== 执行蓝绿部署 ==="
        # 创建绿色环境
        kubectl apply -f k8s/deployment-green.yaml -n ${namespace}
        kubectl wait --for=condition=available deployment/${env.APP_NAME}-green -n ${namespace}
        
        # 切换流量
        kubectl patch service ${env.APP_NAME} -p '{"spec":{"selector":{"version":"green"}}}' -n ${namespace}
        
        # 清理蓝色环境
        kubectl delete deployment ${env.APP_NAME}-blue -n ${namespace} || true
    """
}

/**
 * 金丝雀部署
 */
def canaryDeploy(namespace) {
    sh """
        echo "=== 执行金丝雀部署 ==="
        # 部署金丝雀版本（10%流量）
        kubectl apply -f k8s/deployment-canary.yaml -n ${namespace}
        kubectl wait --for=condition=available deployment/${env.APP_NAME}-canary -n ${namespace}
        
        # 监控5分钟
        sleep 300
        
        # 如果没有问题，完全切换
        kubectl set image deployment/${env.APP_NAME} \
            ${env.APP_NAME}=${env.DOCKER_IMAGE}:${env.DOCKER_TAG} \
            -n ${namespace}
        
        # 清理金丝雀版本
        kubectl delete deployment ${env.APP_NAME}-canary -n ${namespace}
    """
}

/**
 * 根据环境获取命名空间
 */
def getNamespaceByEnv(env) {
    switch(env) {
        case 'dev': return env.K8S_NAMESPACE_DEV
        case 'test': return env.K8S_NAMESPACE_TEST
        case 'prod': return env.K8S_NAMESPACE_PROD
        default: return env.K8S_NAMESPACE_DEV
    }
}

/**
 * 根据环境获取服务URL
 */
def getServiceUrl(env) {
    switch(env) {
        case 'dev': return 'http://dev.company.com'
        case 'test': return 'http://test.company.com'
        case 'prod': return 'http://api.company.com'
        default: return 'http://localhost:8080'
    }
}

/**
 * 发送构建通知
 */
def sendNotification(buildResult) {
    def color = buildResult == 'SUCCESS' ? 'good' : 'danger'
    def emoji = buildResult == 'SUCCESS' ? '✅' : '❌'
    
    // Slack通知
    slackSend(
        channel: env.SLACK_CHANNEL,
        color: color,
        message: """
            ${emoji} 构建${buildResult == 'SUCCESS' ? '成功' : '失败'}
            
            项目: ${env.APP_NAME}
            分支: ${env.BRANCH_NAME}
            构建号: ${env.BUILD_NUMBER}
            环境: ${params.DEPLOY_ENV}
            提交: ${env.GIT_COMMIT_SHORT} by ${env.GIT_AUTHOR}
            
            查看详情: ${env.BUILD_URL}
        """
    )
    
    // 邮件通知
    emailext(
        to: env.EMAIL_RECIPIENTS,
        subject: "${emoji} ${env.APP_NAME} 构建${buildResult == 'SUCCESS' ? '成功' : '失败'} - #${env.BUILD_NUMBER}",
        body: """
            <h2>构建${buildResult == 'SUCCESS' ? '成功' : '失败'}</h2>
            
            <table border="1" cellpadding="5">
                <tr><td>项目名称</td><td>${env.APP_NAME}</td></tr>
                <tr><td>构建分支</td><td>${env.BRANCH_NAME}</td></tr>
                <tr><td>构建编号</td><td>${env.BUILD_NUMBER}</td></tr>
                <tr><td>部署环境</td><td>${params.DEPLOY_ENV}</td></tr>
                <tr><td>Git提交</td><td>${env.GIT_COMMIT_SHORT}</td></tr>
                <tr><td>提交作者</td><td>${env.GIT_AUTHOR}</td></tr>
                <tr><td>提交信息</td><td>${env.GIT_MESSAGE}</td></tr>
                <tr><td>构建时间</td><td>${new Date()}</td></tr>
            </table>
            
            <p><a href="${env.BUILD_URL}">查看构建详情</a></p>
            <p><a href="${env.BUILD_URL}console">查看构建日志</a></p>
        """,
        mimeType: 'text/html'
    )
}