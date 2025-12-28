# [Linux 操作系统面试问答](https://www.nowcoder.com/discuss/770071726645334016?sourceSSR=users)

### 1、什么是Linux操作系统?

​        Linux是一种开源、免费的类Unix操作系统内核，由Linus Torvalds于1991年创建。基于该内核构建的完整操作系统称为Linux发行版（如Ubuntu、CentOS、Debian等），广泛用于服务器、嵌入式设备和个人电脑。

### 2、如何创建一个文件?

[复制代码](#)

```
touch` `filename.txt   ``# 创建空文件``echo` `"content"` `> ``file`  `# 创建带内容的文件``vim filename      ``# 使用文本编辑器创建
```

### 3、如何创建一个文件目录?

[复制代码](#)

```
mkdir` `dirname`     `# 创建单级目录``mkdir` `-p parent``/child` `# 创建多级目录（递归创建）
```

### 4、如何删除一个目录以及目录中所有文件?

[复制代码](#)

```
rm` `-rf ``dirname`    `# 强制递归删除（谨慎使用！）
```

### 5、使用什么命令查看ip地址?

[复制代码](#)

```
ip addr show     ``# 推荐（现代Linux）``ifconfig`       `# 传统命令（部分系统需安装net-tools）
```

### 6、如何重命名一个文件?

[复制代码](#)

```
mv` `oldname.txt newname.txt  ``# 重命名``mv` `file``.txt ``/new/path/`    `# 移动文件
```

### 7、什么是ROOT帐户?

超级用户账户，拥有系统最高权限（可修改任何文件、安装软件、管理用户）。

UID（用户ID）为 0。

- 提示符通常以 # 结尾（普通用户是 $）。

### 8、在a目录下找出大小超过1MB的文件?

[复制代码](#)

```
find` `a/ -``type` `f -size +1M  ``# 查找大于1MB的文件
```

### 9、在a目录中找出，带有test的文件?

[复制代码](#)

```
find` `a/ -``type` `f -name ``"*test*"`  `# 按文件名匹配``grep` `-r ``"test"` `a/        ``# 按文件内容匹配
```

### 10、在Linux下如何查看隐藏文件?

[复制代码](#)

```
ls` `-a    ``# 显示所有文件（包括以`.`开头的隐藏文件）``ls` `-la    ``# 显示详细列表（含权限、大小等）
```

### 11、如何查看 Linux磁盘空间使用情况?

[复制代码](#)

```
df` `-h    ``# 查看所有磁盘分区（人类可读格式）``du` `-sh ``dir`  `# 查看目录占用空间（-s: 总计, -h: 易读格式）
```

### 12、详细说一说VI命令?

- 三种模式：
- 命令模式（默认）：移动光标、复制/粘贴（yy复制行，p粘贴）。
- 插入模式（按 i/a）：编辑文本。
- 末行模式（按 :）：保存、退出、搜索（:wq保存退出，/text搜索）。
- 常用操作：
- dd 删除当前行
- :set number 显示行号
- :q! 强制退出不保存

### 13、如何查看一个文件的权限?

[复制代码](#)

```
ls` `-l filename   ``# 显示权限（如 `-rwxr--r--`）``stat filename    ``# 详细属性（含权限数字码）
```

### 14、如何给一个文件赋予权限?

[复制代码](#)

```
chmod` `u+x ``file`   `# 给所有者添加执行权限``chmod` `g-w ``file`   `# 删除所属组的写权限``chmod` `o=r ``file`   `# 设置其他用户只读
```

### 15、赋权命令Chmod 777，三个数字分别代表什么意思?

三个数字分别代表：

第一个7：所有者权限 rwx (4+2+1=7)

第二个7：所属组权限 rwx

第三个7：其他用户权限 rwx

权限数字对照：

4 = 读（r）

2 = 写（w）

1 = 执行（x）

### 16、在Linux 下如何解压缩?

[复制代码](#)

```
# 解压.tar.gz``tar` `-xzvf ``file``.``tar``.gz    ` `# 解压.zip``unzip ``file``.zip        ` `# 解压.tar.xz``tar` `-xJvf ``file``.``tar``.xz    
```

### 17、如何查看JAVA进程，并关闭进程?

[复制代码](#)

```
ps` `-ef | ``grep` `java     ``# 查找JAVA进程（显示PID）``kill` `-9 PID         ``# 强制终止进程（PID替换为实际ID）``pkill -f ``"java.*arg"`    `# 按名称/参数终止
```

### 18、如何搭建JDK环境?

1）下载JDK压缩包（如 jdk-21_linux-x64.tar.gz）

2）解压到目录（如 /opt/jdk-21）

3）配置环境变量（编辑 ~/.bashrc 或 /etc/profile）

[复制代码](#)

```
export` `JAVA_HOME=``/opt/jdk-21``export` `PATH=$JAVA_HOME``/bin``:$PATH
```

4）生效配置：

[复制代码](#)

```
source` `~/.bashrc
```

5）验证：

[复制代码](#)

```
java -version
```

### 19、如何搭建Tomcat环境?

1）下载Tomcat（如 apache-tomcat-10.x.tar.gz）

2）解压到目录（如 /opt/tomcat）

3）启动：

[复制代码](#)

```
/opt/tomcat/bin/startup``.sh
```

4）验证：浏览器访问 http://服务器IP:8080

### 20、如何搭建 MySQL环境?

[复制代码](#)

```
sudo` `apt update``sudo` `apt ``install` `mysql-server``sudo` `systemctl start mysql``sudo` `mysql_secure_installation ``# 安全配置（设置密码等）
```

[复制代码](#)

```
sudo` `yum ``install` `mysql-server``sudo` `systemctl start mysqld``sudo` `mysql_secure_installation
```

[复制代码](#)

```
mysql -u root -p  ``# 登录``SHOW DATABASES;  ``# 查看数据库
```

作者：玖拾肆
链接：https://www.nowcoder.com/discuss/770071726645334016?sourceSSR=users
来源：牛客网