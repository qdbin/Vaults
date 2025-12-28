# [SQL语句面试问答（一）](https://www.nowcoder.com/discuss/770311746505412608?sourceSSR=users)

### 一、单表查询面试题：

[复制代码](#)

```
CREATE TABLE students (``  ``studentNo ``int``(``10``) primary key auto_increment, ``  ``name varchar(``10``), ``  ``sex varchar(``1``), ``  ``hometown varchar(``20``), ``  ``age ``int``(``4``), ``  ``class` `varchar(``10``), ``  ``card varchar(``20``)``)；` `INSERT INTO students (name, sex, hometown, age, ``class``, card)``VALUES(``'王一'``,``'男'``,``'北京'``,``'20'``,``'1班'``,``'123456'``), ``  ``(``'诸二'``,``'男'``,``'上海'``,``'18'``,``'2班'``,``'123789'``), ``  ``(``'张三'``,``'男'``,``'南京'``,``'124'``,``'3班'``,``'123147'``), ``  ``(``'白四'``,``'男'``,``'安徽'``,``'22'``,``'4班'``,``'123258'``), ``  ``(``'杨五'``,``'女'``,``'天津'``,``'19'``,``'3班'``,``'123369'``), ``  ``(``'孙六'``,``'女'``,``'河北'``,``'18'``,``'1班'``,``'145236'``), ``  ``(``'赵七'``,``'男'``,``'山西'``,``'20'``,``'2班'``,``'125698'``), ``  ``(``'小八'``,``'女'``,``'河南'``,``'15'``,``'3班'``,NULL), ``  ``(``'久久'``,``'男'``,``'湖南'``,``'21'``,``'1班'``,``''``), ``  ``(``'史十'``,``'女'``,``'广东'``,``'26'``,``'2班'``,``'159875'``), ``  ``(``'李十一'``,``'男'``,``'北京'``,``'30'``,``'4班'``,``'147896'``), ``  ``(``'孙十二'``,``'男'``,``'新疆'``,``'26'``,``'3班'``,``'125632'``);`` ` `CREATE TABLE courses (``courseNo ``int``(``10``) PRIMARY KEY AUTO_INCREMENT, ``name varchar(``10``)``);`` ` `INSERT INTO courses``VALUES (``'1'``,``'数据库'``), ``  ``(``'2'``,``'qtp'``), ``  ``(``'3'``,``'Linux'``), ``  ``(``'4'``,``'系统测试'``), ``  ``(``'5'``,``'单元测试'``), ``  ``(``'6'``,``'测试过程'``);` `CREATE TABLE scores (``  ``id ``int``(``10``) PRIMARY KEY AUTO_INCREMENT, ``  ``courseNo ``int``(``10``), ``  ``studentNo ``int``(``10``), ``  ``score ``int``(``4``)``)；` `INSERT INTO scores``VALUES (``'1'``,``'1'``,``1``,``'90'``), ``  ``(``'2'``,``'2'``,``2``,``'98'``), ``  ``(``'3'``,``'1'``,``2``,``'75'``), ``  ``(``'4'``,``'3'``,``1``,``'86'``), ``  ``(``'5'``,``'3'``,``3``,``'80'``), ``  ``(``'6'``,``'4'``,``4``,``'79'``), ``  ``(``'7'``,``'5'``,``5``,``'96'``), ``  ``(``'8'``,``'6'``,``6``,``'80'``);
```

#### 1、查询学生"张三”的基本信息

[复制代码](#)

```
select * from students where name=``'张三'
```

#### 2、查询学生“李十一“或“孙十二”的基本信息

[复制代码](#)

```
select * from students where name=``'李十一'` `or name=``'孙十二'
```

#### 3、查询姓“张”学生的姓名，年龄，班级

[复制代码](#)

```
select name,age,``class` `from students where name=``'张%'
```

#### 4、查询姓名中含有“一”字的学生的基本信息

[复制代码](#)

```
select * from student where name=``'%一%'
```

#### 5、查询姓名长度为三个字，姓“孙”的学生的学号，姓名，年龄，班级，身份证号

[复制代码](#)

```
select studentNo，name, age, ``class``, card from students where name like ``'孙__'` `select studentNo，name, age, ``class``, card from students ``where char_length(name) = ``3` `-- 确保姓名长度为``3``个字符``  ``and 姓名 like ``'孙%'``;  -- 确保姓``"孙"
```

#### 6、查询姓“白”或者姓“孙”的学生的基本信息

[复制代码](#)

```
select * from students where name like ``'白%'` `or name like ``'孙%'
```

#### 7、查询姓"白"并且家乡是"山西”的学生信息

[复制代码](#)

```
select * from students where name like ``'白%'` `and hometown=``'山西'
```

#### 8、查询家乡是“北京”、“新疆”、“山东"或者"上海"的学生的信息

[复制代码](#)

```
select * from students where hometown=``'北京'` `or hometown=``'新疆'` `or hometown=``'山东'` `or hometown=``'上海'` `select * from students where hometown in (``'北京'``,``'新疆'``,``'山东'``,``'上海'``)
```

#### 9、查询姓“孙”，但是家乡不是“河北”的学生信息

[复制代码](#)

```
select * from students where name like ``'孙%'` `and honmetown!=``'河北'
```

#### 10、查询家乡不是“北京”、“新疆”、"山东”、”上海”的学生的信息

[复制代码](#)

```
select * from students ``where hometown!=``'北京'` `or hometown!=``'新疆'` `or hometown!=``'山东'` `or hometown!=``'上海'` `select * from students where hometown not in (``'北京'``,``'新疆'``,``'山东'``,``'上海'``)
```

#### 11、查询全部学生信息，并按照“性别”排序

[复制代码](#)

```
select * from students order by sex
```

#### 12、查询现有学生都来自于哪些不同的省份

[复制代码](#)

```
select honetown from students group by hometown
```

#### 13、查询所有男生，并按年龄升序排序

[复制代码](#)

```
select sex from students where sex=``'男'` `order by age asc
```

#### 14、统计共有多少个学生

[复制代码](#)

```
select count(*) from students 
```

#### 15、统计年龄大于20岁的学生有多少个

[复制代码](#)

```
select count(*) from students where age>``'20'
```

#### 16、统计男生的平均年龄

[复制代码](#)

```
select avg(age) from students where sex=``'男'
```

#### 17、查询1班学生中的最大年龄是多少

[复制代码](#)

```
select max(age) from students where ``class``=``'1班'
```

#### 18、统计2班男女生各有多少人

[复制代码](#)

```
select sum(sex=``'男'``) as ``'男生总数'``,sum(sex=``'女'``) as ``'女生总数'` `from students where ``class``=``'2班'` `select sex,count(*) from students where ``class``=``'2班'` `group by sex
```

#### 19、统计每个班级中每种性别的学生人数，并按照班级升序排序

[复制代码](#)

```
select ``class``,sex,count(*) from students group by ``class``,sex order by ``class
```

问题来源：米兔软件测试

作者：玖拾肆
链接：https://www.nowcoder.com/discuss/770311746505412608?sourceSSR=users
来源：牛客网

# [SQL语句面试问答（二）](https://www.nowcoder.com/discuss/771023657618395136?sourceSSR=users)

### 二、多表查询面试题

[复制代码](#)

```
create` `table` `departments (``deptid ``int``(10) ``primary` `key``, ``deptname ``varchar``(20) ``not` `null` `-- 部门名称``)；``insert` `into` `departments ``values` `(``'1001'``,``'市场部'``)；``insert` `into` `departments ``values` `(``'1002'``,``'测试部'``)；``insert` `into` `departments ``values` `(``'1003'``,``'开发部'``)；` `create` `table` `employees (``  ``empid ``int``(10) ``primary` `key``, ``  ``empname ``varchar``(20) ``not` `null``,``-- 姓名``  ``sex ``varchar``(4) ``default` `null``, ``-- 性别``  ``deptid ``int``(20) ``default` `null``,``-- 部门编号``  ``jobs ``varchar``(20) ``default` `null``, ``-- 岗位``  ``politicalstatus ``varchar``(20) ``default` `null``,``-- 政治面貌``  ``leader ``int``(10) ``default` `null``)；` `insert` `into` `employees ``values` `(``'1'``,``'王一'``,``'女'``,``'1003'``,``'开发'``,``'群众'``,``'9'``)；``insert` `into` `employees ``values` `(``'2'``,``'朱二'``,``'男'``,``'1003'``,``'开发经理'``,``'群众'``,``null``)；``insert` `into` `employees ``values` `(``'3'``,``'张三'``,``'男'``,``'1002'``,``'测试'``,``'团员'``,``'4'``)；``insert` `into` `employees ``values` `(``'4'``,``'白四'``,``'男'``,``'1002'``,``'测试经理'``,``'党员'``,``null``)；``insert` `into` `employees ``values` `(``'5'``,``'杨五'``,``'女'``,``'1002'``,``'测试'``,``'党员'``,``'4'``);``insert` `into` `employees ``values` `(``'6'``,``'孙六'``,``'女'``,``'1001'``,``'市场'``,``'党员'``,``'12'``);``insert` `into` `employees ``values` `(``'7'``,``'赵七'``,``'男'``,``'1001'``,``'市场'``,``'团员'``,``'12'``);``insert` `into` `employees ``values` `(``'8'``,``'小八'``,``'女'``,``'1002'``,``'测试'``,``'群众'``,``'4'``);``insert` `into` `employees ``values` `(``'9'``,``'久久'``,``'男'``,``'1003'``,``'开发'``,``'党员'``,``'9'``);``insert` `into` `employees ``values` `(``'10'``,``'史十'``,``'女'``,``'1003'``,``'开发'``,``'团员'``,``'9'``)；``insert` `into` `employees ``values` `(``'11'``,``'李十一'``,``'男'``,``'1002'``,``'测试'``,``'团员'``,``'4'``)；``insert` `into` `employees ``values` `(``'12'``,``'孙十二'``,``'男'``,``'1001'``,``'市场经理'``,``'党员'``,``null``)；` `create` `table` `salary (``  ``sid ``int``(10) ``primary` `key``, ``  ``empid ``int``(10) ``not` `nul1, ``  ``salary ``int``(10) ``not` `null``-- 工资``)；` `insert` `into` `salary ``values` `(``'1'``,``'7'``,``'2100'``);``insert` `into` `salary ``values` `(``'2'``,``'6'``,``'2000'``)；``insert` `into` `salary ``values` `(``'3'``,``'12'``,``'5000'``)；``insert` `into` `salary ``values` `(``'4'``,``'9'``,``'1999'``)；``insert` `into` `salary ``values` `(``'5'``,``'10'``,``'1900'``);``insert` `into` `salary ``values` `(``'6'``,``'1'``,``'3000'``);``insert` `into` `salary ``values` `(``'7'``,``'2'``,``'5500'``);``insert` `into` `salary ``values` `(``'8'``,``'5'``,``'2000'``)；``insert` `into` `salary ``values` `(``'9'``,``'3'``,``'1500'``)；``insert` `into` `salary ``values` `(``'10'``,``'8'``,``'4000'``)；``insert` `into` `salary ``values` `(``'11'``,``'11'``,``'2600'``)；``insert` `into` `salary ``values` `(``'12'``,``'4'``,``'5300'``)；
```

#### 1.列出总人数大于4的部门号和总人数

[复制代码](#)

```
select` `deptid， ``count``(*) ``from` `employees ``group` `by` `deptid ``having` `count``(*)>4
```

#### 2.列出开发部和测试部的职工号，姓名

[复制代码](#)

```
select` `e.empid,d.deptname ``from` `employees ``as` `e ``inner` `join` `deptnames ``as` `d ``on` `e.empid=d.deptid ``where` `d.department ``in` `(``'开发部'``,``'测试部'``)
```

#### 3.求出各部门党员的人数，要求显示部门名称

[复制代码](#)

```
select` `d.deptname , ``count``(*)``from` `employees ``as` `e ``inner` `join` `departments ``as` `d ``on` `e.empid=d.deptid ``where` `e.politicalstatus=``'党员'` `group` `by` `e.deptid
```

#### 4.列出市场部的所有女职工的姓名和政治面貌

[复制代码](#)

```
select` `e.empname，e.politicalstatus ``from` `employees ``as` `e ``inner` `join` `departments ``as` `d ``on` `e.empid=d.deptid``where` `e.sex=``'女'` `and` `d.deptname=``'市场部'
```

#### 5.显示所有职工的姓名，部门名和工资数

[复制代码](#)

```
select` `e.empname,d.deptname,s.salary``from` `employees ``as` `e ``inner` `join` `departments ``as` `d ``on` `e.empid=d.deptid``inner` `join` `salary ``as` `s ``on` `e.empid=s.empid
```

#### 6.显示各部门名和该部门的职工平均工资

[复制代码](#)

```
select` `d.deptname, ``avg``(s.salary) ``from` `departments ``as` `d ``inner` `join` `employees ``as` `e ``on` `e.empid=d.deptid``inner` `join` `salary ``as` `s ``on` `e.empid=s.empid ``group` `by` `d.deptname
```

#### 7.显示工资最高的前3名职工的职工号和姓名

[复制代码](#)

```
select` `e.empid,e.empname,s.salary ``from` `employees ``as` `e``inner` `join` `salary ``as` `s ``on` `e.empid=s.empid``order` `by` `s.salary ``desc` `limit 3
```

#### 8.列出工资在1000-2000之间的所有职工姓名

[复制代码](#)

```
select` `e.empname,s.salary ``from` `salary ``as` `s``inner` `join` `employees ``as` `e ``on` `e.empid=s.empid``where` `s.salary ``between` `1000 ``and` `2000
```

#### 9.列出工资比王一高的员工

[复制代码](#)

```
select` `* ``from` `employees ``as` `e``inner` `join` `salary ``as` `s ``on` `e.empid=s.empid``where` `s.salary > (``select` `s.salary ``from` `employees ``as` `e ``inner` `join` `salary ``as` `s ``on` `e.empid=s.empid ``where` `e.empname=``'王一'``)
```

作者：玖拾肆
链接：https://www.nowcoder.com/discuss/771023657618395136?sourceSSR=users
来源：牛客网

# [SQL语句面试问答（三）](https://www.nowcoder.com/discuss/771360740941778944?sourceSSR=users)

[复制代码](#)

```
DROP` `TABLE` `IF EXISTS ``'students'``;``CREATE` `TABLE` `'students'` `(``  ``'Sid'` `varchar``(11) ``NOT` `NULL``, ``-- 学生ID ``  ``'Sname'` `varchar``(20) ``DEFAULT` `NULL``, ``-- 学生名字``  ``'Sage'` `int``(11) ``DEFAULT` `NULL``, ``-- 学生年龄``  ``PRIMARY` `KEY` `(``'Sid'``)``) ENGINE=InnoDB ``DEFAULT` `CHARSET=utf8;` `INSERT` `INTO` `'students'` `VALUES` `(``'s01'``,``'巧虎'``,``'18'``)；``INSERT` `INTO` `'students'` `VALUES` `(``'s02'``,``'琪琪'``,``'19'``)；``INSERT` `INTO` `'students'` `VALUES` `(``'s03'``,``'桃乐比'``,``'18'``)；``INSERT` `INTO` `'students'` `VALUES` `(``'s04'``,``'圆圆'``,``'17'``);``INSERT` `INTO` `'students'` `VALUES` `(``'s05'``,``'美美'``,``'20'``);``INSERT` `INTO` `'students'` `VALUES` `(``'s06'``,``'多比'``,``'25'``);
```

[复制代码](#)

```
DROP` `TABLE` `IF EXISTS ``'teachers'``;``CREATE` `TABLE` `'teachers'` `(``  ``'Tid'` `varchar``(11) ``NOT` `NULL``, ``-- 老师id``  ``'Tname'` `varchar``(20) ``DEFAULT` `NULL``, ``-- 老师名字``  ``PRIMARY` `KEY` `(``'Tid'``)``) ENGINE=InnoDB ``DEFAULT` `CHARSET=utf8;` `INSERT` `INTO` `'teachers'` `VALUES` `(``'1'``,``'汪老师'``)；``INSERT` `INTO` `'teachers'` `VALUES` `(``'2'``,``'石老师'``)；``INSERT` `INTO` `'teachers'` `VALUES` `(``'3'``,``'李老师'``)；``INSERT` `INTO` `'teachers'` `VALUES` `(``'4'``,``'熊老师'``)；``INSERT` `INTO` `'teachers'` `VALUES` `(``'5'``,``'王老师'``)；``INSERT` `INTO` `'teachers'` `VALUES` `(``'6'``,``'梁老师'``)；
```

[复制代码](#)

```
DROP` `TABLE` `IF EXISTS ``'courses'``;``CREATE` `TABLE` `'courses'``(``  ``'Cid'` `varchar``(11) ``NOT` `NULL``, ``-- 课程id``  ``'Cname'` `varchar``(20) ``DEFAULT` `NULL``, ``-- 课程名字``  ``'Cteacher'` `varchar``(20) ``DEFAULT` `NULL``, ``-- 课程老师``  ``PRIMARY` `KEY` `(``'Cid'``)``)ENGINE=InnoDB ``DEFAULT` `CHARSET=utf8;` `INSERT` `INTO` `'courses'` `VALUES` `(``'c01'``,``'美术'``,``'汪老师'``)；``INSERT` `INTO` `'courses'` `VALUES` `(``'c02'``,``'音乐'``,``'石老师'``)；``INSERT` `INTO` `'courses'` `VALUES` `(``'c03'``,``'体能'``,``'李老师'``)；``INSERT` `INTO` `'courses'` `VALUES` `(``'C04'``,``'自行车'``,``'熊老师'``)；``INSERT` `INTO` `'courses'` `VALUES` `(``'c05'``,``'钢琴'``,``'王老师'``)；
```

[复制代码](#)

```
DROP` `TABLE` `IF EXISTS ``'results'``;``CREATE` `TABLE` `'results'``(``  ``'id'` `int``(11) ``NOT` `NULL` `AUTO_INCREMENT, ``-- 成绩id``  ``'Sid'` `varchar``(11) ``DEFAULT` `NULL``, ``-- 学生 id``  ``'Cid'` `varchar``(11) ``DEFAULT` `NULL``, ``-- 课程 id``  ``'Cscore'` `varchar``(11) ``DEFAULT` `NULL``, ``-- 课程成绩``  ``PRIMARY` `KEY` `(``'id'``)``) ENGINE=InnoDB AUTO_INCREMENT=16 ``DEFAULT` `CHARSET=utf8;` `INSERT` `INTO` `'results'` `VALUES` `(``'1'``,``'s01'``,``'c01'``,``'58'``);``INSERT` `INTO` `'results'` `VALUES` `(``'2'``,``'s01'``,``'c02'``,``'59'``);``INSERT` `INTO` `'results'` `VALUES` `(``'3'``,``'s01'``,``'c03'``,``'55'``)；``INSERT` `INTO` `'results'` `VALUES` `(``'4'``,``'s02'``,``'c02'``,``'83'``);``INSERT` `INTO` `'results'` `VALUES` `(``'5'``,``'s02'``,``'c05'``,``'79'``);``INSERT` `INTO` `'results'` `VALUES` `(``'6'``,``'s02'``,``'c04'``,``'77'``)；``INSERT` `INTO` `'results'` `VALUES` `(``'7'``,``'s03'``,``'c01'``,``'55'``)；``INSERT` `INTO` `'results'` `VALUES` `(``'8'``,``'s03'``,``'c03'``,``'81'``)；``INSERT` `INTO` `'results'` `VALUES` `(``'9'``,``'s03'``,``'c04'``,``'73'``);``INSERT` `INTO` `'results'` `VALUES` `(``'10'``,``'s04'``,``'c01'``,``'67'``);``INSERT` `INTO` `'results'` `VALUES` `(``'11'``,``'s04'``,``'c02'``,``'78'``)；``INSERT` `INTO` `'results'` `VALUES` `(``'12'``,``'s04'``,``'c03'``,``'82'``)；``INSERT` `INTO` `'results'` `VALUES` `(``'13'``,``'s04'``,``'c05'``,``'80'``);``INSERT` `INTO` `'results'` `VALUES` `(``'14'``,``'s05'``,``'c01'``,``'61'``);``INSERT` `INTO` `'results'` `VALUES` `(``'15'``,``'s04'``,``'c04'``,``'78'``);
```

#### 1、查询并统计同龄学生人数

[复制代码](#)

```
select` `Sage ``as` `年龄,``count``(*) ``as` `人数 ``from` `students ``group` `by` `Sage ``order` `by` `Sage
```

#### 2、查询每门课程的平均成绩，结果按平均成绩升序排列，平均成绩相同时，按课程号降序排列

[复制代码](#)

```
select` `Cid, ``avg``(CScore) ``from` `results ``group` `by` `Cid ``order` `by` `avg``(CScore) ,Cid ``desc
```

#### 3、按平均成绩从高到低显示所有学生的平均成绩

[复制代码](#)

```
select` `Sid, ``avg``(CScore) ``from` `results ``group` `by` `Sid ``order` `by` `avg``(Cscore) ``desc
```

#### 4、查询各科成绩最高分、最低分和平均分：以如下形式显示：课程ID，课程name，最高分，最低分，平均分

[复制代码](#)

```
select` `r.Cid ``as` `课程ID, c.Cname ``as` `课程``name``, ``max``(r.Cscore) ``as` `最高分, ``min``(r.Cscore) ``as` `最低分, ``avg``(r.Cscore) ``as` `平均分``from` `result ``as` `r ``inner` `join` `courses ``as` `c ``on` `r.cid=c.cid``group` `by` `课程ID
```

#### 5、按各科平均成绩从低到高顺序排序

[复制代码](#)

```
select` `Cid, ``avg``(Cscore) ``from` `results ``group` `by` `Cid
```

#### 6、查询学生的总成绩并进行排名

[复制代码](#)

```
select` `Sid， ``sum``(Cscore) ``from` `results ``group` `by` `Sid ``order` `by` `sum``(Cscore) ``desc
```

#### 7、检索至少选修两门课程的学生学号

[复制代码](#)

```
select` `Sid， ``count``(Cid) ``from` `results ``order` `by` `Sid ``having` `count``(Cid)>=2
```

#### 8、查询出只选修了一门课程的全部学生的学号和姓名

[复制代码](#)

```
select` `r.Sid,s.Sname ``from` `student ``as` `s ``inner` `join` `result ``as` `r ``where` `s.Sid=r.Sid``group` `by` `r.Sid ``having` `count``(r.Cid)=1
```

#### 9、查询每门课都在70分以上的学生id

[复制代码](#)

```
select` `Sid ``from` `results ``group` `by` `Sid ``having` `min``(Cscore)>70
```

#### 10、查询课程编号为"c01"且课程成绩在60分以上的学生的学号和姓名

[复制代码](#)

```
select` `r.Sid, s.Sname ``from` `students ``as` `s ``inner` `join` `results ``as` `r ``where` `s.Sid=r.Sid``where` `r.sid=``'c01'` `and` `r.Cscore>=60
```

#### 11、查询平均成绩大于60分的同学的学号和平均成绩

[复制代码](#)

```
select` `Sid, ``avg``(Cscore) ``from` `results ``group` `by` `Sid ``having` `avg``(Cscore)>60
```

#### 12、查询所有同学的学号、姓名、选课数、总成绩

[复制代码](#)

```
select` `r.Sid,s.Sname,``count``(r.Cid),``sum``(r.Cscore) ``from` `students ``as` `s ``inner` `join` `results ``as` `r ``where` `s.Sid=r.Sid``group` `by` `r.Sid
```

作者：玖拾肆
链接：https://www.nowcoder.com/discuss/771360740941778944?sourceSSR=users
来源：牛客网

# [SQL语句面试问答（四）](https://www.nowcoder.com/discuss/773310033860796416?sourceSSR=users)

## 一、单表查询

### 1、基础查询

#### 1）查询所有列

[复制代码](#)

```
SELECT` `* ``FROM` `TableName;
```

#### 2）查询特定列

[复制代码](#)

```
SELECT` `Column1, Column2 ``FROM` `TableName;
```

#### 3） 列别名

[复制代码](#)

```
SELECT` `Column1 ``AS` `name1, Column2 ``AS` `name2 ``FROM` `TableName;
```

#### 4） 去重查询

[复制代码](#)

```
SELECT` `DISTINCT` `Column1 ``FROM` `TableName;
```

#### 5）限制返回行数

[复制代码](#)

```
SELECT` `FROM` `TableName LIMIT 10;
```

#### 6）分页查询

[复制代码](#)

```
SELECT` `FROM` `TableName ``LIMIT 10 ``-- 限制返回结果的数量为 10 条记录``OFFSET 20; ``-- 跳过查询结果的前 20 条记录
```

#### 7） 排序查询

[复制代码](#)

```
SELECT` `* ``FROM` `TableName ``ORDER` `BY` `Column1 ``DESC``;
```

#### 8） 多列排序

[复制代码](#)

```
SELECT` `* ``FROM` `TableName ``ORDER` `BY` `Column1 ``DESC``, Column2 ``ASC``;``-- DESC 排序从大到小，ASC 排序从小到大
```

### 2、数据过滤

#### 1） 基础过滤

[复制代码](#)

```
SELECT` `* ``FROM` `TableName ``WHERE` `Column1 > value1; ``-- >, <, >=, <=，!=，=
```

#### 2） 多条件过滤

[复制代码](#)

```
SELECT` `* ``FROM` `TableName ``WHERE` `Column1 > value1 ``AND` `Column2 > value2;``SELECT` `* ``FROM` `TableName ``WHERE` `Column1 > value1 ``OR` `Column2 > value2;
```

#### 3）范围查询

[复制代码](#)

```
SELECT` `* ``FROM` `TableName ``WHERE` `Column1 ``BETWEEN` `value1 ``AND` `value2；
```

#### 4） IN操作符

[复制代码](#)

```
SELECT` `* ``FROM` `TableName ``WHERE` `Column1 ``IN` `(value1, value2，value3)； 
```

#### 5）模糊查询

[复制代码](#)

```
SELECT` `* ``FROM` `TableName ``WHERE` `Column1 ``LIKE` `'%value%'``;``-- 匹配值中间为value``SELECT` `* ``FROM` `TableName ``WHERE` `Column1 ``LIKE` `'%value'``;``-- 匹配值以value结尾``SELECT` `* ``FROM` `TableName ``WHERE` `Column1 ``LIKE` `'value%'``;``-- 匹配值以value开头
```

#### 6） NULL值判断

[复制代码](#)

```
SELECT` `* ``FROM` `TableName ``WHERE` `Column1 ``IS` `NULL``;      
```

#### 7）排除特定值

[复制代码](#)

```
SELECT` `* ``FROM` `TableName ``WHERE` `Column1 != value;
```

### 3、聚合函数

#### 1） 计算总数

[复制代码](#)

```
SELECT` `COUNT``(*) ``AS` `cnt ``FROM` `TableName ``WHERE` `column1 = value;
```

#### 2）分组求和

[复制代码](#)

```
SELECT` `column1 ``AS` `col1. ``SUM``(column2) ``AS` `col2 ``FROM` `TableName ``GROUP` `BY` `column1;
```

#### 3）分组平均值

[复制代码](#)

```
SELECT` `column1 ``AS` `col1, ``AVG``(column2) ``AS` `col2 ``FROM` `TableName ``GROUP` `BY` `column1;
```

#### 4） 分组最大值

[复制代码](#)

```
SELECT` `column1 ``AS` `col1, ``MAX``(column2) ``AS` `col2 ``FROM` `TableName ``GROUP` `BY` `column1;
```

#### 5） 分组最小值

[复制代码](#)

```
SELECT` `column1 ``AS` `col1, ``MIN``(column2) ``AS` `co12 ``FROM` `TableName ``GROUP` `BY` `column1;
```

#### 6） 分组筛选(HAVING)

[复制代码](#)

```
SELECT` `column1 ``AS` `col1, ``SUM``(column2) ``AS` `col2 ``FROM` `TableName      ``WHERE` `column3=value ``HAVING` `SUM``(column2) > value;
```

#### 7） 多列分组

[复制代码](#)

```
SELECT` `column1 ``AS` `col1, column1 ``AS` `col2, ``SUM``(column3) ``AS` `col3 ``FROM` `TableName``GROUP` `BY` `column1, column2;
```

### 4、高级窗口函数

#### 1）ROW_NUMBER 生成唯一序号

[复制代码](#)

```
SELECT` `column1, column2, ROW_NUMBER() OVER (``ORDER` `BY` `column2) ``AS` `row ``FROM` `TableName;
```

#### 2） RANK 与 DENSE_RANK 排名

[复制代码](#)

```
SELECT` `column1, column2, RANK() OVER (``ORDER` `BY` `column2 ``DESC``) ``AS` `rank, ``DENSE_RANK() OVER (``ORDER` `BY` `column2 ``DESC``) ``AS` `dense_rank ``FROM` `TableName；
```

#### 3） 累计百分比计算

[复制代码](#)

```
SELECT` `column1, column2, ``SUM``(column2) OVER (``ORDER` `BY` `column1) / ``SUM``(column2) ``OVER() ``AS` `cumulative_percent ``FROM` `TableName；
```

#### 4）移动平均（最近三个窗口）

[复制代码](#)

```
SELECT` `column1, column2,``AVG``(column2) OVER (``ORDER` `BY` `column1 ``ROWS` `BETWEEN` `2``PRECEDING ``AND` `CURRENT` `ROW) ``AS` `moving_avg ``FROM` `TableName;
```

#### 5）分组内前N名

[复制代码](#)

```
SELECT` `* ``FROM`` ``(``SELECT` `column1, column2, column3, ROW_NUMBER() OVER (PARTITION ``BY` `column1 ``ORDER` `BY``column2 ``DESC``) ``AS` `rn ``FROM` `TableName) ``WHERE` `In` `<= 3；
```

## 二、多表查询

### 1、表连接操作

#### 1） 内连接

[复制代码](#)

```
SELECT` `t1.column1, t2.column2 ``FROM` `Table1 t1``JOIN` `Table2 t2 ``ON` `t1.column3 = t2.column3；
```

#### 2） 左连接

[复制代码](#)

```
SELECT` `t1.column1, t2.column2 ``FROM` `Table1 t1``LEFT` `JOIN` `Table2 t2 ``ON` `t1.column3 = t2.column3;
```

#### 3） 右连接

[复制代码](#)

```
SELECT` `t1.column1, t2.column2 ``FROM` `Table1 t1``RIGHT` `JOIN` `Table2 t2 ``ON` `t1.column3 = t2.column3;
```

#### 4） 全外连接

[复制代码](#)

```
SELECT` `t1.column1, t2.column2 ``FROM` `Table1 t1``FULL` `OUTER` `JOIN` `Table2 t2 ``ON` `t1.column3 = t2.column3;
```

#### 5） 自连接

[复制代码](#)

```
SELECT` `t1.``column` `as` `column1, t2.``column` `as` `column2``FROM` `Table1 t1 ``JOIN` `Table1 t2 ``ON` `t1.column1=t2.column2;
```

#### 6） 交叉连接

[复制代码](#)

```
SELECT` `* ``FROM` `Colors ``CROSS` `JOIN` `Sizes;
```

### 2、子查询

#### 1）标量子查询

[复制代码](#)

```
SELECT` `column1, (``SELECT` `COUNT``(*) ``FROM` `TableB ``WHERE` `column2= a.column2) ``AS` `cnt ``FROM` `TableA a;
```

#### 2） IN子查询

[复制代码](#)

```
SELECT` `column1 ``FROM` `TableA``WHERE` `column2 ``IN``(``SELECT` `column2 ``FROM` `Categories ``WHERE` `Name``= ``'xxxxxxx'``);
```

#### 3）EXISTS子查询

[复制代码](#)

```
SELECT` `column1 ``FROM` `TableA a ``WHERE` `EXISTS (``SELECT` `1 ``FROM` `TableB ``WHERE` `column2 = a.column2)；
```

#### 4）**子查询作为派生表**

[复制代码](#)

```
SELECT` `AVG``(``sum``) ``AS` `avg` `FROM` `(``SELECT` `SUM``(column2) ``AS` `sum` `FROM` `TABLEA ``GROUP` `BY` `column1) ``AS` `t;
```

#### 5）**多条件子查询**

[复制代码](#)

```
SELECT` `column1, column2 ``FROM` `TableA ``WHERE` `column2 > (``SELECT` `AVG``(column2) ``FROM` `TableA);
```

### 3、联合查询部分

#### 1）**去重联合(**`UNION`（去重）**)**

[复制代码](#)

```
SELECT` `column1 ``FROM` `TableA ``UNION` `SELECT` `column1 ``FROM` `TableB;
```

#### **2）不去重联合(**`UNION ALL`（保留重复）**)**

[复制代码](#)

```
SELECT` `column1 ``FROM` `TableA ``UNION` `ALL` `SELECT` `column1 ``FROM` `TableB;
```

## 三、常用函数

### 1、字符串处理

#### 1）**字符串长度**

[复制代码](#)

```
SELECT` `LENGTH(column1) ``FROM` `TableName;
```

#### 2）**字符串截取（SUBSTRING(字符串, 起始位置, 截取长度)）**

[复制代码](#)

```
SELECT` `SUBSTRING` `(, start, length) ``FROM` `TableName;
```

#### 3）**字符串替换**

[复制代码](#)

```
SELECT` `REPLACE``(column1, ``'old_string'``, ``'new_string'``) ``FROM` `TableName;
```

#### 4）**字符串拼接**

[复制代码](#)

```
SELECT` `CONCAT(column1, column2) ``FROM` `TableName;
```

#### 5）字符串转大写

[复制代码](#)

```
SELECT` `UPPER``(column_name) ``FROM` `TableName;
```

#### 6）**字符串转小写**

[复制代码](#)

```
SELECT` `LOWER``(column_name) ``FROM` `TableName;
```

### 2、时间日期函数

#### 1）**当前时间**

[复制代码](#)

```
SELECT` `CURTIME();
```

#### **2）当前日期**

[复制代码](#)

```
SELECT` `CURDATE();
```

#### **3）当前日期和时间**

[复制代码](#)

```
SELECT` `NOW();
```

#### **4）日期向后加天数**

[复制代码](#)

```
SELECT` `DATE_ADD(NOW(), INTERVAL 10 ``DAY``);
```

#### **5）日期减天数**

[复制代码](#)

```
SELECT` `DATE_SUB(NOW(), INTERVAL 10 ``DAY``);
```

#### **6）获取两个日期差值**

[复制代码](#)

```
SELECT` `DATEDIFF(date1, date2);
```

#### **7）获取日期年份**

[复制代码](#)

```
SELECT` `YEAR``(``date``) ``FROM` `TableName;
```

#### **8）获取月份**

[复制代码](#)

```
SELECT` `MONTH``(``date``) ``FROM` `TableName;
```

#### **9）获取日**

[复制代码](#)

```
SELECT` `DAY``(``date``) ``FROM` `TableName;
```

#### 10）**获取小时**

[复制代码](#)

```
SELECT` `HOUR``(time_column) ``FROM` `TableName;
```

#### **11）获取分钟**

[复制代码](#)

```
SELECT` `MINUTE``(time_column) ``FROM` `TableName;
```

#### **12）获取秒**

[复制代码](#)

```
SELECT` `SECOND``(time_column) ``FROM` `TableName;
```

#### **13）获取周数（一年中的第几周）**

[复制代码](#)

```
SELECT` `WEEK(date_column) ``FROM` `TableName;``-- 可添加模式参数：WEEK(date_column, 0)（0-周日开始，1-周一开始）
```

#### **14）日期转字符串**

[复制代码](#)

```
SELECT` `DATE_FORMAT(date_column, ``'%Y-%m-%d'``) ``FROM` `TableName;` `-- 常用格式：``-- %Y：4位年份``-- %m：月份（01-12）``-- %d：日（01-31）``-- %H:%i:%s：时分秒
```

#### **15）字符串转日期**

[复制代码](#)

```
SELECT` `CAST``(string_column ``AS` `DATE``) ``FROM` `TableName;` `SELECT` `STR_TO_DATE(``'2023-01-15'``, ``'%Y-%m-%d'``);
```

## **四、常用操作**

### **1、数据操作**

#### **1）插入单条数据**

[复制代码](#)

```
INSERT` `INTO` `TableName (Column1, Column2) ``VALUES` `(value1, value2);
```

#### **2）插入多条数据（批量插入）**

[复制代码](#)

```
INSERT` `INTO` `TableName (Column1, Column2) ``VALUES` `(value1, value2), (value3, value4);
```

#### **3）更新数据（带条件）**

[复制代码](#)

```
UPDATE` `TableName ``SET` `Column1 = value1 ``WHERE` `Column2 = value2;
```

#### **4）条件删除数据**

[复制代码](#)

```
DELETE` `FROM` `Orders ``WHERE` `OrderDate < ``'2020-01-01'``;
```

#### **5）全表删除（保留表结构）**

[复制代码](#)

```
DELETE` `FROM` `TempData;
```

#### **6）清空表数据（高效重置）**

[复制代码](#)

```
TRUNCATE` `TABLE` `Logs;``-- 特点：不可回滚，重置自增计数器，不触发DELETE触发器*
```

### **2、表操作**

#### **1）创建新表**

[复制代码](#)

```
CREATE` `TABLE` `TableName (``  ``column1 ``INT` `PRIMARY` `KEY``,``  ``column2 ``VARCHAR``(50),``  ``column3 ``DATE``);
```

#### **2）添加新列**

[复制代码](#)

```
ALTER` `TABLE` `TableName ``ADD` `COLUMN` `new_column ``INT``;
```

#### **3）修改列类型**

[复制代码](#)

```
ALTER` `TABLE` `TableName ``MODIFY` `COLUMN` `column1 ``VARCHAR``(20);``-- MySQL语法，其他数据库可能使用 ALTER COLUMN
```

#### **4）删除列**

[复制代码](#)

```
ALTER` `TABLE` `TableName ``DROP` `COLUMN` `column1;
```

#### **5）重命名表**

[复制代码](#)

```
ALTER` `TABLE` `TableName RENAME ``TO` `NewTableName;``-- SQL Server使用 sp_rename，Oracle使用 RENAME
```

#### **6）删除表**

[复制代码](#)

```
DROP` `TABLE` `TableName;``-- 将同时删除表结构和数据
```

### **3、约束与索引**

#### **1）添加主键约束**

[复制代码](#)

```
ALTER` `TABLE` `TableName ``ADD` `PRIMARY` `KEY` `(column1);
```

#### **2）添加唯一约束**

[复制代码](#)

```
ALTER` `TABLE` `TableName ``ADD` `UNIQUE` `(column1);
```

#### **3）添加外键约束（补充）**

[复制代码](#)

```
ALTER` `TABLE` `Orders ``ADD` `CONSTRAINT` `fk_customer``FOREIGN` `KEY` `(customer_id) ``REFERENCES` `Customers(id);``-- 注：确保被引用列（column2）是主键或唯一键
```

#### **4）创建索引**

[复制代码](#)

```
CREATE` `INDEX` `idx_column ``ON` `TableName (column1);` `-- 扩展为复合索引``CREATE` `INDEX` `idx_multi ``ON` `TableName (column1, column2);
```

#### 5）删除索引

[复制代码](#)

```
DROP` `INDEX` `idx_column1 ``ON` `TableName;` `-- 不同数据库语法差异：``-- PostgreSQL/Oracle: DROP INDEX idx_column1;``-- SQL Server: DROP INDEX TableName.idx_column1;
```

#### 6）设置非空约束

[复制代码](#)

```
ALTER` `TABLE` `TableName ``MODIFY` `COLUMN` `column1 ``VARCHAR``(100) ``NOT` `NULL``;
```

#### 7）移除非空约束

[复制代码](#)

```
ALTER` `TABLE` `TableName ``MODIFY` `COLUMN` `column1 ``VARCHAR``(100) ``NULL``;
```

### **4、视图**

#### **1）创建视图**

[复制代码](#)

```
CREATE` `VIEW` `ViewName ``AS` `SELECT` `column1, column2 ``FROM` `TableName ``WHERE` `condition;` `-- 添加计算列``CREATE` `VIEW` `SalesSummary ``AS``SELECT` `product_id, ``SUM``(quantity) ``AS` `total_qty ``FROM` `Orders ``GROUP` `BY` `product_id;
```

#### **2）通过视图更新数据**

[复制代码](#)

```
UPDATE` `ViewName ``SET` `column1 = ``'value'` `WHERE` `condition;``-- 限制：视图必须满足可更新条件（不包含聚合、DISTINCT等）
```

#### **3）删除视图**

[复制代码](#)

```
DROP` `VIEW` `IF EXISTS ViewName;
```

### **5、事务控制**

#### **1）开启事务**

[复制代码](#)

```
START ``TRANSACTION``;` `-- 使用数据库特定语法：``-- SQL Server: BEGIN TRANSACTION``-- Oracle: SET TRANSACTION
```

#### **2）提交事务**

[复制代码](#)

```
COMMIT``;``-- 确认所有操作永久生效
```

#### **3）回滚事务**

[复制代码](#)

```
ROLLBACK``;``-- 撤销事务内所有未提交的操作
```

#### **4）设置保存点**

[复制代码](#)

```
SAVEPOINT savepoint1;``-- 在事务中创建回滚标记点
```

#### **5）回滚到保存点**

[复制代码](#)

```
ROLLBACK` `TO` `savepoint1;``-- 撤销保存点之后的操作，保留之前的操作
```

### 6、权限管理

#### 1）**授予查询权限**

[复制代码](#)

```
-- 允许user1读取指定表数据``GRANT` `SELECT` `ON` `TableName ``TO` `user1;
```

#### **2）授予所有权限**

[复制代码](#)

```
-- 权限范围包括：SELECT, INSERT, UPDATE, DELETE等``GRANT` `ALL` `PRIVILEGES` `ON` `DatabaseName.* ``TO` `'admin'``@``'localhost'``;
```

#### **3）撤销权限**

[复制代码](#)

```
-- 移除user2对指定表的删除权限``REVOKE` `DELETE` `ON` `TableName ``FROM` `user2;
```

### 7、其他操作

#### 1）**列出所有数据库**

[复制代码](#)

```
SHOW DATABASES;` `-- MySQL语法，其他数据库等效命令：``-- SQL Server: SELECT name FROM sys.databases``-- PostgreSQL: \l (psql命令行)``-- Oracle: SELECT * FROM v$database
```

#### **2）列出当前数据库所有表**

[复制代码](#)

```
SHOW TABLES;` `-- 查看指定数据库的表：``SHOW TABLES ``FROM` `database_name;
```

#### **3）查看表结构**

[复制代码](#)

```
DESCRIBE TableName;` `-- 等效命令：``-- MySQL: DESC TableName``-- SQL Server: sp_help 'TableName'``-- PostgreSQL: \d TableName
```

#### **4）查看建表语句**

[复制代码](#)

```
SHOW ``CREATE` `TABLE` `TableName;``-- 输出结果包含完整DDL语句，可用于表重建
```

#### **5）查询表的所有列**

[复制代码](#)

```
SELECT` `COLUMN_NAME ``FROM` `INFORMATION_SCHEMA.COLUMNS``WHERE` `TABLE_SCHEMA = ``'database_name'`` ``AND` `TABLE_NAME = ``'table_name'``;
```

#### **6）查看表索引信息**

[复制代码](#)

```
SHOW ``INDEX` `FROM` `TableName;` `-- 输出字段说明：``-- Non_unique: 是否唯一索引 (0=唯一, 1=非唯一)``-- Key_name: 索引名称``-- Seq_in_index: 索引中的列序号``-- Column_name: 索引列名
```

#### **7）查询表存储大小**

[复制代码](#)

```
SELECT`` ``table_name ``AS` `'Table'``,`` ``ROUND((data_length + index_length) / 1024 / 1024, 2) ``AS` `'Size (MB)'` `-- 修正原图的LEMETH拼写错误``FROM` `information_schema.TABLES``WHERE` `table_schema = ``'database_name'``;` `-- (data_length + index_length) -- 数据大小 + 索引大小（字节）``-- / 1024 / 1024         -- 转换为MB``-- ROUND(..., 2)         -- 保留2位小数` `-- SQL Server查看表大小``EXEC` `sp_spaceused ``'TableName'``;` `-- PostgreSQL查看表大小``SELECT` `pg_size_pretty(pg_total_relation_size(``'TableName'``));
```

#### **8）设置会话时区**

[复制代码](#)

```
SET` `time_zone = ``'Asia/Shanghai'``;` `SET` `time_zone = ``'+8:00'``; ``-- 东八区偏移量表示
```

#### **9）创建数据库**

[复制代码](#)

```
CREATE` `DATABASE` `database_name ``CHARACTER` `SET` `utf8mb4 ``COLLATE` `utf8mb4_unicode_ci;` `CREATE` `DATABASE` `inventory ``DEFAULT` `CHARACTER` `SET` `utf8mb4``DEFAULT` `COLLATE` `utf8mb4_0900_ai_ci; ``-- MySQL 8.0+推荐排序规则
```

#### **10）删除数据库**

[复制代码](#)

```
DROP` `DATABASE` `IF EXISTS database_name;` `-- Oracle等效命令：``DROP` `USER` `schema_name ``CASCADE``; ``-- Oracle中数据库用户即schema
```

作者：玖拾肆
链接：https://www.nowcoder.com/discuss/773310033860796416?sourceSSR=users
来源：牛客网