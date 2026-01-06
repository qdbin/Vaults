# [aiohttp中ClientSession使用注意事项](https://www.cnblogs.com/lymmurrain/p/13805690.html)

最近在研究协程，想写个协程实现的爬虫，选用aiohttp，对aiohttp中 ClientSession使用有些不解,然而中文资料有点少，大多是写怎么用就没了，不是很详细，就直接看英文官网了。

aiohttp可用作客户端与服务端，写爬虫的话用客户端即可，所以本文只关于aiohttp的客户端使用(发请求)，并且需要一点协程的知识才能看懂。

如果想要研究aiohttp的话推荐直接看英文官网，写的很通俗易懂，就算不大懂英文，直接翻译也能看懂七八成了。

以下参考自https://docs.aiohttp.org/en/stable/，如有纰漏，欢迎斧正。

### 简单请求

如果只发出简单的请求(如只有一次请求，无需cookie，SSL，等)，可用如下方法。

但其实吧很少用，因为一般爬虫中用协程都是要爬取大量页面，可能会使得aiohttp报Unclosed client session的错误。这种情况官方是建议用ClientSession(连接池，见下文)的，性能也有一定的提高。

```python
import aiohttp

async def fetch():
    async with aiohttp.request('GET',
            'http://python.org/') as resp:
        assert resp.status == 200
        print(await resp.text())
#将协程放入时间循环        
loop = asyncio.get_event_loop()
loop.run_until_complete(fetch())     
```

### 使用连接池请求

一般情况下使用如下示例,由官网摘抄。

```python
import aiohttp
import asyncio

#传入client使用
async def fetch(client,url):
    async with client.get(url) as resp:
        assert resp.status == 200
        return await resp.text()

async def main():
    async with aiohttp.ClientSession() as client:
        html = await fetch(client,url)
        print(html)

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
```

是不是感觉有点绕呢，其实平时使用是不必这样将fetch函数抽象出去，可以简单写成下面的简洁示例。

```python
import aiohttp
import asyncio
async def main():
    async with aiohttp.ClientSession() as client:
        async with aiohttp.request('GET',
                'http://python.org/') as resp:
            assert resp.status == 200
            print(await resp.text())
```

发现有什么不同没有，官网的fetch函数抽象出去后，把ClientSession的一个实例作为参数了。所以在with代码块中使用ClientSession实例的情况下，这两者是等同的(我认为，因为两者都是用的都是with代码块中创建的实例)。

#### 连接池重用

而其实官网这段代码是在ClientSession的参考处摘抄的，所以官方这样写我认为只是在提醒要注意ClientSession的用法。那么ClientSession有啥得注意的呢

> Session 封装了一个*连接池*（*连接器*实例），并且默认情况下支持keepalive。除非在应用程序的生存期内连接到大量未知的不同服务器，否则建议您在应用程序的生存期内使用单个会话以受益于连接池。
>
> 不要为每个请求创建Session 。每个应用程序很可能需要一个会话，以完全执行所有请求。
>
> 更复杂的情况可能需要在每个站点上进行一次会话，例如，一个会话用于Github，另一个会话用于Facebook API。无论如何，为每个请求建立会话是一个**非常糟糕的**主意。
>
> 会话内部包含一个连接池。连接重用和保持活动状态（默认情况下均处于启用状态）可能会提高整体性能。

以上这几段话由官网翻译而来。这几段话都是说，如无必要，只用一个ClientSession实例即可。

但我在很多资料看到的是像如下这样用的呀

```python
async def fetch(url):
    async with aiohttp.ClientSession() as client:
        async with aiohttp.request('GET',
                url) as resp:
            assert resp.status == 200
            print(await resp.text())
    
```

这不明显没请求一次就实例化一个ClientSession嘛，并没有重用ClientSession啊。那应该咋办呢，然而官网并没有举出重用ClientSession的示例(我也是服了，你这么浓墨重彩说道只需一个session，倒是给个示例啊)。

那只得继续找找资料。然而国内资料不多，只能上github和stackoverflow看看。看了半天也没个定论，主要是两个方法。

##### 在with代码块中用一个session完成所有请求

下面是我写的示例

```python
async def fetch(client,url):
    async with client.get(url) as resp:
        assert resp.status == 200
        text = await resp.text()
        return len(text)

#urls是包含多个url的列表
async def fetch_all(urls):
    async with aiohttp.ClientSession() as client:
        return await asyncio.gather(*[fetch(client,url) for url in urls])
    
urls = ['http://python.org/' for i in range(3)]
loop=asyncio.get_event_loop()
results = loop.run_until_complete(fetch_all(urls))
print(results)
print(type(results))
```

##### 手动创建session，不用with

该方法可以让你获取一个session实例而不仅局限于with代码块中，可以在后续代码中继续使用该session。

```python
async def fetch(client,url):
    async with client.get(url) as resp:
        assert resp.status == 200
        text = await resp.text()
        return len(text)

async def fetch_all_manual(urls,client):
    return await asyncio.gather(*[fetch(client, url) for url in urls])

urls = ['http://python.org/' for i in range(3)]
loop=asyncio.get_event_loop()
client = aiohttp.ClientSession()
results = loop.run_until_complete(fetch_all_manual(urls,client))
#要手动关闭自己创建的ClientSession，并且client.close()是个协程，得用事件循环关闭
loop.run_until_complete(client.close())
#在关闭loop之前要给aiohttp一点时间关闭ClientSession
loop.run_until_complete(asyncio.sleep(3))
loop.close()
print(results)
print(type(results))
```

此处着重说明以下该方法一些相关事项

- 手动创建ClientSession要手动关闭自己创建的ClientSession，并且client.close()是个协程，得用事件循环关闭。
- 在关闭loop之前要给aiohttp一点时间关闭ClientSession

如果无上述步骤会报Unclosed client session的错误，也即ClientSession没有关闭

但就算你遵循了以上两个事项，如此运行程序会报以下warning，虽然不会影响程序正常进行

```vbnet
DeprecationWarning: The object should be created from async function
  client = aiohttp.ClientSession()
```

这说的是`client = aiohttp.ClientSession()`这行代码应该在异步函数中执行。如果你无法忍受可以在定义个用异步方法用作创建session

```python
async def create_session():
    return aiohttp.ClientSession()

session = asyncio.get_event_loop().run_until_complete(create_session())
```

#### ClientSession 部分重要参数

下面是ClientSession的所有参数，这里用的比较多的是connector,headers,cookies。headers和cookies写过爬虫的可能都认识了，这里只谈一下connector。

connector是aiohttp客户端API的传输工具。并发量控制，ssl证书验证，都可通过connector设置，然后传入ClientSession。

标准connector有两种：

1. [`TCPConnector`](https://docs.aiohttp.org/en/stable/client_reference.html#aiohttp.TCPConnector)用于常规*TCP套接字*（同时支持*HTTP*和 *HTTPS*方案）(绝大部分情况使用这种)。
2. [`UnixConnector`](https://docs.aiohttp.org/en/stable/client_reference.html#aiohttp.UnixConnector) 用于通过UNIX套接字进行连接（主要用于测试）。

所有连接器类都应继承自[`BaseConnector`](https://docs.aiohttp.org/en/stable/client_reference.html#aiohttp.BaseConnector)。

使用可以按以下实例

```python
#创建一个TCPConnector
conn=aiohttp.TCPConnector(verify_ssl=False)
#作为参数传入ClientSession
async with aiohttp.ClientSession(connector=conn) as session: 
```

TCPConnector比较重要的参数有

- **verify_ssl**（[*bool*](https://docs.python.org/3/library/functions.html#bool)）–布尔值，对*HTTPS*请求执行SSL证书验证 （默认情况下启用）。当要跳过对具有无效证书的站点的验证时可设置为False。
- **limit**（[*int*](https://docs.python.org/3/library/functions.html#int)）–整型，同时连接的总数。如果为*limit*为 `None`则connector没有限制（默认值：100）。
- **limit_per_host**（[*int*](https://docs.python.org/3/library/functions.html#int)）–限制同时连接到同一端点的总数。如果`(host, port, is_ssl)`三者相同，则端点是相同的。如果为*limit=0*，则connector没有限制（默认值：0）。

如果爬虫用上协程，请求速度是非常快的，很可能会对别人服务器造成拒绝服务的攻击，所以平常使用若无需求，最好还是不要设置**limit**为0。

##### 限制并发量的另一个做法(使用Semaphore)

使用Semaphore直接限制发送请求。此处只写用法，作抛砖引玉之用。也很容易用，在fetch_all_manual函数里加上Semaphore的使用即可

```python
async def fetch(client,url):
    async with client.get(url) as resp:
        assert resp.status == 200
        text = await resp.text()
        return len(text)

async def fetch_all_manual(urls,client):
    async with asyncio.Semaphore(5):
        return await asyncio.gather(*[fetch(client, url) for url in urls])

sem
urls = ['http://python.org/' for i in range(3)]
loop=asyncio.get_event_loop()
client = aiohttp.ClientSession()
results = loop.run_until_complete(fetch_all_manual(urls,client))
#要手动关闭自己创建的ClientSession，并且client.close()是个协程，得用事件循环关闭
loop.run_until_complete(client.close())
#在关闭loop之前要给aiohttp一点时间关闭ClientSession
loop.run_until_complete(asyncio.sleep(3))
loop.close()
print(results)
print(type(results))
```

#### 参考文献

https://www.cnblogs.com/wukai66/p/12632680.html

https://stackoverflow.com/questions/46991562/how-to-reuse-aiohttp-clientsession-pool

https://stackoverflow.com/questions/35196974/aiohttp-set-maximum-number-of-requests-per-second/43857526#43857526

https://github.com/aio-libs/aiohttp/issues/4932

https://www.cnblogs.com/c-x-a/p/9248906.html