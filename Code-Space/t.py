import functools


def retry(times: int):
    """重试装饰器，当函数抛出异常时自动重试指定次数。

    Args:
        times: 最大重试次数（含首次调用）。
    """
    if times < 1:
        raise ValueError("times 必须 >= 1")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < times:
                        print(f"[retry] {func.__name__}() 第 {attempt}/{times} 次失败，" f"异常: {e!r}，准备重试...")
            raise last_exception

        return wrapper

    return decorator


# ---- 测试 ----
if __name__ == "__main__":
    # 示例 1：基本重试
    @retry(times=3)
    def test():
        print("call")
        raise RuntimeError("error")

    try:
        test()
    except RuntimeError as e:
        print(f"最终失败: {e}")

    print("-" * 40)

    # 示例 2：带参数的成功调用
    @retry(times=5)
    def add(a, b):
        return a + b

    result = add(1, 2)
    print(f"add(1, 2) = {result}")

    print("-" * 40)

    # 示例 3：第三次才成功
    call_count = 0

    @retry(times=4)
    def flaky():
        global call_count
        call_count += 1
        print(f"第 {call_count} 次调用")
        if call_count < 3:
            raise ValueError("还没准备好")
        return "成功了！"

    print(flaky())
