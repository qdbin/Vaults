## 开篇

![image-20251015052415964](assets/image-20251015052415964.png)

## 缓存

### 缓存穿透

![image-20251015053751116](assets/image-20251015053751116.png)

> ![image-20251015054316272](assets/image-20251015054316272.png)

### 缓存击穿

<img src="./assets/image-20251015055406878.png" alt="image-20251015055406878" style="zoom:50%;" />

<img src="./assets/image-20251015055503693.png" alt="image-20251015055503693" style="zoom:67%;" />



> ![image-20251015054653501](assets/image-20251015054653501.png)
>
> ![image-20251015055251698](assets/image-20251015055251698.png)

### 缓存雪崩

<img src="./assets/image-20251015060107157.png" alt="image-20251015060107157" style="zoom:43%;" />

> ![image-20251015055919409](assets/image-20251015055919409.png)

### 数据持久化

![image-20251016034501537](assets/image-20251016034501537.png)

![image-20251016034518644](assets/image-20251016034518644.png)

> ![image-20251016044012710](assets/image-20251016044012710.png)
>
> ![image-20251016033749891](assets/image-20251016033749891.png)
>
> <img src="./assets/image-20251016034338759.png" alt="image-20251016034338759" style="zoom:60%;" />

### mysql如何与redis进行同步？

<img src="./assets/image-20251016032012138.png" alt="image-20251016032012138" style="zoom:70%;" />

<img src="./assets/image-20251015060225430.png" alt="image-20251015060225430" style="zoom:33%;" />

> #### 双写一致
>
> ![image-20251016030735279](assets/image-20251016030735279.png)
>
> ![image-20251016031320870](assets/image-20251016031320870.png)
>
> ![image-20251016031446672](assets/image-20251016031446672.png)
>
> ![image-20251016031632573](assets/image-20251016031632573.png)

### Redis的key过期后，会立即删除吗？（数据过期策略）

<img src="./assets/image-20251016035655105.png" alt="image-20251016035655105" style="zoom:67%;" />

> ![image-20251016035505614](assets/image-20251016035505614.png)

### 假如缓存过多，内存是有限的，内存被占满了怎么办(数据淘汰策略)？

<img src="./assets/image-20251016040509316.png" alt="image-20251016040509316" style="zoom:53%;" />



<img src="./assets/image-20251016040347961.png" alt="image-20251016040347961" style="zoom:40%;" />

> ![image-20251016040234327](assets/image-20251016040234327.png)
>
> ![image-20251016040241949](assets/image-20251016040241949.png)