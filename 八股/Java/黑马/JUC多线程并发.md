# [JUC多线程并发](https://www.bilibili.com/video/BV1yT411H7YK?p=86&vd_source=a3edfa2588cde0dbe42d7192afacee48)

## 线程基础

### 线程和进程的区别

![image-20251022182652818](./assets/image-20251022182652818.png)

### 并行和并发的区别

<img src="./assets/image-20251022182847593.png" alt="image-20251022182847593" style="zoom:50%;" />

### 创建线程的方式

<img src="./assets/image-20251022183148374.png" alt="image-20251022183148374" style="zoom:50%;" />

### 线程6种状态

<img src="./assets/image-20251022183350982.png" alt="image-20251022183350982" style="zoom:50%;" />

### 线程执行顺序

<img src="./assets/image-20251022183607341.png" alt="image-20251022183607341" style="zoom:50%;" />

### notify（）和notifyAll（）区别

<img src="./assets/image-20251022183701497.png" alt="image-20251022183701497" style="zoom:50%;" />

### sleep（）和wait（）方法

![image-20251022184039093](./assets/image-20251022184039093.png)

### 停止正在运行的线程

![image-20251022190321436](./assets/image-20251022190321436.png)

## 并发安全

### synchronized关键字底层原理

<img src="./assets/image-20251022191510894.png" alt="image-20251022191510894" style="zoom:60%;" />

> ![image-20251022191534323](./assets/image-20251022191534323.png)

### 锁升级（Monitor实现的锁属于重量级锁）

![image-20251022200110115](./assets/image-20251022200110115.png)

> #### 对象怎么关联Monitor
>
> ![image-20251022202550313](./assets/image-20251022202550313.png)
>
> ![image-20251022202710760](./assets/image-20251022202710760.png)
>
> ![image-20251022202814899](./assets/image-20251022202814899.png)

###  JMM（Java内存模型）

<img src="./assets/image-20251022203601934.png" alt="image-20251022203601934" style="zoom:57%;" />

> ![image-20251022203644683](./assets/image-20251022203644683.png)

### CAS

<img src="./assets/image-20251022211535589.png" alt="image-20251022211535589" style="zoom:50%;" />

> #### CAS数据交换流程
>
> <img src="./assets/image-20251022211910925.png" alt="image-20251022211910925" style="zoom:35%;" />
>
> #### CAS底层实现
>
> <img src="./assets/image-20251022212006056.png" alt="image-20251022212006056" style="zoom:40%;" />
>
> #### CAS乐观锁
>
> ![image-20251022212049223](./assets/image-20251022212049223.png)

### Volatile

<img src="./assets/image-20251022214319952.png" alt="image-20251022214319952" style="zoom:50%;" />

> #### 可见性
>
> ![image-20251022213016118](./assets/image-20251022213016118.png)
>
> #### 禁止指令重排
>
> ![image-20251022214450742](./assets/image-20251022214450742.png)

### AQS（锁机制-抽象队列同步器）

<img src="./assets/image-20251022220124918.png" alt="image-20251022220124918" style="zoom:50%;" />

> #### 说明
>
> <img src="./assets/image-20251022215538313.png" alt="image-20251022215538313" style="zoom:33%;" />
>
> #### 工作机制
>
> ![image-20251022215939802](./assets/image-20251022215939802.png)

### ReentrantLock的实现原理（未看）

<img src="./assets/image-20251022220314642.png" alt="image-20251022220314642" style="zoom:53%;" />

### Synchronized和Lock有什么区别？

<img src="./assets/image-20251023013548069.png" alt="image-20251023013548069" style="zoom: 37%;" />

> #### 可打断
>
> <img src="./assets/image-20251023014623775.png" alt="image-20251023014623775" style="zoom: 50%;" />
>
> #### 可超时
>
> <img src="./assets/image-20251023015157574.png" alt="image-20251023015157574" style="zoom: 50%;" />
>
> #### 多条件变量
>
> <img src="./assets/image-20251023013926495.png" alt="image-20251023013926495" style="zoom:67%;" />

### 死锁

<img src="./assets/image-20251023020930105.png" alt="image-20251023020930105" style="zoom:50%;" />



> ![image-20251023020505245](./assets/image-20251023020505245.png)

### ConcurrentHashMap

> 1. **JDK 1.8** 的**CAS自旋保证了head_node添加元素的原子性**（当对应下标**插槽没有head_node**（即没有数据），**可以避免**对应**对应插槽**多个线程同时**添加数据的并发安全问题**），添加成功后就给head_node添加SynchronizLock
> 2. **JDK 1.7** 的**分段数组不能扩容**，其HashEntry数组可以扩容；多个Key定位同一个segment下标数组，由于有**ReentrantLock**,故**只能有一个线程操作HashEntry数组数据**；

<img src="./assets/image-20251023023104732.png" alt="image-20251023023104732" style="zoom:50%;" />

> #### JDK 1.8
>
> <img src="./assets/image-20251023023152801.png" alt="image-20251023023152801" style="zoom:60%;" />
>
> #### JDK 1.7
>
> <img src="./assets/image-20251023023240308.png" alt="image-20251023023240308" style="zoom:60%;" />

### 并发问题的根本原因（Java怎么保证线程安全）

<img src="./assets/image-20251023025215833.png" alt="image-20251023025215833" style="zoom:70%;" />

## 线程池

### 核心7大参数（ThreadPoolExecutor）

<img src="./assets/image-20251023030602333.png" alt="image-20251023030602333" style="zoom:45%;" />

### 线程池执行原理

![image-20251023030641034](./assets/image-20251023030641034.png)

### 常见阻塞队列

![image-20251023031609253](./assets/image-20251023031609253.png)

> **数组阻塞队列 & 链表阻塞队列 ** **区别**
>
> ![image-20251023031543597](./assets/image-20251023031543597.png)

### 确认核心线程数？

![image-20251023033500713](./assets/image-20251023033500713.png)

> #### IO密集&CPU密集
>
> ![image-20251023033748635](./assets/image-20251023033748635.png)

### 线程池种类

<img src="./assets/image-20251023034248588.png" alt="image-20251023034248588" style="zoom:55%;" />

> #### 固定线程数线程池
>
> <img src="./assets/image-20251023035925488.png" alt="image-20251023035925488" style="zoom:50%;" />
>
> #### 单线程化线程池
>
> ![image-20251023040404694](./assets/image-20251023040404694.png)
>
> #### 可缓存线程池
>
> ![image-20251023040323385](./assets/image-20251023040323385.png)
>
> #### 执行延迟任务的线程池
>
> ![image-20251023041159746](./assets/image-20251023041159746.png)

### 为什么不建议用Executors创建线程池（未看）



## 使用场景（未看）

### 你们的项目哪里用到了多线程（三个场景示例）？

### 如何控制某个方法运行并发访问线程的数量？

## 其他

### 谈谈你对ThreadLocal的理解（底层未了解）

<img src="./assets/image-20251023035552869.png" alt="image-20251023035552869" style="zoom:60%;" />