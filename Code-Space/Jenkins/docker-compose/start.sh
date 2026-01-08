#!/bin/bash

# 定义变量
BASE_HOME=/root/app                    # 应用根目录
JAR_NAME=gitlab-demo-1.0.jar          # jar包文件名
LOG_NAME=app.log                       # 日志文件名

# 杀死正在运行的Java进程
ps -ef | grep $JAR_NAME | grep -v grep | awk '{print $2}' | xargs -i kill {}

# 备份旧的日志文件（如果存在）
if [ -f $BASE_HOME/$LOG_NAME ]; then 
    mv $BASE_HOME/$LOG_NAME $BASE_HOME/$LOG_NAME.`date +%Y%m%d%H%M%S` 
fi 

# 备份旧的jar包（如果存在）
if [ -f $BASE_HOME/$JAR_NAME ]; then 
    cp $BASE_HOME/$JAR_NAME $BASE_HOME/$JAR_NAME.`date +%Y%m%d%H%M%S` 
fi 

# 启动新的Java应用
nohup java -jar $BASE_HOME/$JAR_NAME &>$BASE_HOME/$LOG_NAME &