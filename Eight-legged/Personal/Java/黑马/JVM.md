### 什么是程序计数器？

<img src="./assets/image-20251012205431137.png" alt="image-20251012205431137" style="zoom:50%;" />

### 你能给我详细的介绍java堆吗？

> <img src="./assets/image-20251012210909467.png" alt="image-20251012210909467" style="zoom:47%;" />

**答：**

 <img src="./assets/image-20251012210437444.png" alt="image-20251012210437444" style="zoom:50%;" />

### 虚拟机栈

> #### 答案
>
> <img src="./assets/image-20251012212805304.png" alt="image-20251012212805304" style="zoom:50%;" />

1. #### 什么是虚拟机栈

   <img src="./assets/image-20251012212947065.png" alt="image-20251012212947065" style="zoom:33%;" />

2. #### 方法内的局部变量是否安全？

   <img src="./assets/image-20251012212412109.png" alt="image-20251012212412109" style="zoom:50%;" />

3. #### 栈内存溢出情况

> 递归 or 栈帧过大

<img src="./assets/image-20251012212538697.png" alt="image-20251012212538697" style="zoom:43%;" />

### 方法区/常量池

#### 总结

<img src="./assets/image-20251012215404673.png" alt="image-20251012215404673" style="zoom:50%;" />

#### 方法区

<img src="./assets/image-20251012215514468.png" alt="image-20251012215514468" style="zoom:45%;" />

#### 常量池/运行常量池

![image-20251012215237827](assets/image-20251012215237827.png)

### 直接内存

<img src="./assets/image-20251014014318998.png" alt="image-20251014014318998" style="zoom:50%;" />

> #### 数据拷贝（常规IO/NIO（直接内存））![image-20251014014235361](assets/image-20251014014235361.png)

## 类加载器

### 什么是类加载器，类加载器有哪些？

<img src="./assets/image-20251014014902410.png" alt="image-20251014014902410" style="zoom:45%;" />

> #### 图示：
>
> <img src="./assets/image-20251014014843168.png" alt="image-20251014014843168" style="zoom:45%;" />

### JVM双亲委派

<img src="./assets/image-20251014015500068.png" alt="image-20251014015500068" style="zoom:50%;" />

#### JVM为什么采用双亲委派机制？

![image-20251014015407030](assets/image-20251014015407030.png)

#### 什么是双亲委派模型？

<img src="./assets/image-20251014015247055.png" alt="image-20251014015247055" style="zoom:25%;" />

#### 说一下类装载的执行过程？

![image-20251014023557136](assets/image-20251014023557136.png)

> **加载：**
>
> <img src="./assets/image-20251014023657278.png" alt="image-20251014023657278" style="zoom:33%;" />
>
> **连接**
>
> - **验证**
>
>   <img src="./assets/image-20251014023807656.png" alt="image-20251014023807656" style="zoom:40%;" />
>
> - **准备**
>
>   <img src="./assets/image-20251014023903824.png" alt="image-20251014023903824" style="zoom:40%;" />
>
> - **解析**
>
>   <img src="./assets/image-20251014024000366.png" alt="image-20251014024000366" style="zoom:40%;" />
>
> **初始化**
>
> <img src="./assets/image-20251014024052537.png" alt="image-20251014024052537" style="zoom:40%;" />
>
> **使用**
>
> <img src="./assets/image-20251014024112070.png" alt="image-20251014024112070" style="zoom:40%;" />
>
> 

## 垃圾回收GC

### 对象什么时候可以被垃圾回收？

<img src="./assets/image-20251015034809155.png" alt="image-20251015034809155" style="zoom:50%;" />

> ps：主要回收堆中的对象
>
> #### 引用计数法：
>
> <img src="./assets/image-20251015034246867.png" alt="image-20251015034246867" style="zoom:50%;" />
>
> #### 可达性分析算法
>
> <img src="./assets/image-20251015034731209.png" alt="image-20251015034731209" style="zoom:33%;" />

### 垃圾回收算法有哪些？

<img src="./assets/image-20251015035355332.png" alt="image-20251015035355332" style="zoom:50%;" />

> #### 标记清除
>
> ![image-20251015035046002](assets/image-20251015035046002.png)
>
> #### 标记整理算法
>
> ![image-20251015035209070](assets/image-20251015035209070.png)
>
> #### 复制算法
>
> ![image-20251015035321013](assets/image-20251015035321013.png)

### 说一下JVM中的分代回收

<img src="./assets/image-20251015040543869.png" alt="image-20251015040543869" style="zoom:50%;" />

> #### 分代收集算法-工作机制
>
> ![image-20251015040049443](assets/image-20251015040049443.png)
>
> #### MinorGC、Mixed GC 、 FullGC的区别是什么？
>
> ![image-20251015040421294](assets/image-20251015040421294.png)

### 说一下JVM有哪些垃圾回收器？

<img src="./assets/image-20251015041420164.png" alt="image-20251015041420164" style="zoom:50%;" />

>![image-20251015040943552](assets/image-20251015040943552.png)
>
>![image-20251015041003884](assets/image-20251015041003884.png)
>
>![image-20251015041306015](assets/image-20251015041306015.png)

### 详细聊一下G1垃圾回收器

<img src="./assets/image-20251015044315533.png" alt="image-20251015044315533" style="zoom:50%;" />

> ![image-20251015042235188](assets/image-20251015042235188.png)
>
> #### 三个阶段
>
> ![image-20251015043654996](assets/image-20251015043654996.png)
>
> ![image-20251015043919440](assets/image-20251015043919440.png)
>
> ![image-20251015044224161](assets/image-20251015044224161.png)

### 强，软，弱，虚引用的区别

<img src="./assets/image-20251015050419068.png" alt="image-20251015050419068" style="zoom:50%;" />

>![image-20251015050244213](assets/image-20251015050244213.png)

## JVM调优

### JVM的调优参数可以在哪里设置？

<img src="./assets/image-20251015051120626.png" alt="image-20251015051120626" style="zoom:33%;" />

> ![image-20251015051058212](assets/image-20251015051058212.png)

### JVM的调优参数都有哪些？（没看）

> ![image-20251015051235389](assets/image-20251015051235389.png)

### 说一下JVM调优的工具（没看）

### JVM内存泄漏的排查思路（没看）

### CPU飙高排查方案的思路（没看）
