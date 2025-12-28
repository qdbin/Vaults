## [AOP](https://v.douyin.com/shqKWbRpjZg/)

### 一、核心概念

**1. 定义**\ 

Spring AOP 是面向切面编程的轻量级实现

，通过动态代理在运行时织入横切逻辑（如日志、事务），

无需修改原始代码

。

**2. 核心术语**

- 切面（Aspect）

  ：封装横切逻辑的类（如日志切面）

- 切点（Pointcut）

  ：匹配方法的规则（如

  ```
  execution(* com.service.*(..))
  ```

  ）

- 通知（Advice）

  ：增强逻辑（@Before/@After 等）

- 连接点（JoinPoint）

  ：方法调用点

- 织入（Weaving）：动态插入增强逻辑的过程

- 代理（Proxy）：生成的增强对象

### 二、实现原理

**1. 动态代理机制**

- **JDK 动态代理**

  - 基于接口实现，生成`$Proxy`类，通过反射调用`InvocationHandler.invoke()`

  - 示例：

    ```
    Proxy.newProxyInstance(classLoader, interfaces, (proxy, method, args) -> {  
        // 前置逻辑  
        Object result = method.invoke(target, args);  
        // 后置逻辑  
        return result;  
    });  
    ```

- **CGLIB 代理**

  - 基于类继承，生成目标类的子类
  - 通过`Enhancer`设置父类和回调函数

**2. 选择策略**

- 优先使用 JDK 代理（接口存在时）
- 通过`@EnableAspectJAutoProxy(proxyTargetClass = true)`强制 CGLIB 代理

### 三、关键特性

| 特性            | 说明                             |
| --------------- | -------------------------------- |
| 织入时机        | 运行时动态织入，无需编译期修改   |
| 功能范围        | 仅支持方法级增强                 |
| 与 AspectJ 关系 | 集成 AspectJ 注解，但功能更轻量  |
| 事务支持        | 通过`@Transactional`自动管理事务 |

### 四、面试高频点

**1. 动态代理区别**

| 类型     | 优势             | 局限              |
| -------- | ---------------- | ----------------- |
| JDK 代理 | 接口约束，性能高 | 无法代理类        |
| CGLIB    | 无接口要求       | 无法代理`final`类 |

**2. 常见问题**

- **循环依赖处理**：通过三级缓存延迟代理创建
- **内部方法调用失效**：需通过`AopContext.currentProxy()`获取代理对象

### 五、实战配置

**1. 注解配置示例**

```
@Aspect  
@Component  
public class LoggingAspect {  
    @Pointcut("execution(* com.service.*(..))")  
    public void serviceMethods() {}  

    @Before("serviceMethods()")  
    public void logBefore(JoinPoint joinPoint) {  
        System.out.println("Before method: " + joinPoint.getSignature().getName());  
    }  
}  
```

**2. 启动配置**

```
@Configuration  
@EnableAspectJAutoProxy  
public class AppConfig {  
    // 注册Bean  
}  
```