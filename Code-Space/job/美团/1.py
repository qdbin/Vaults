"""
这个题目是三人行的一道题，题目给定两个数，一个是 N，一个是 X。N 代表从 1 到 N 的数组，X 就是一个给定的值，要求排列后的数组满足：每三个数中间就一定要有一个数比 X 大。

举个例子，假如输入是 5 和 3，那么数组就是 1 2 3 4 5，要求就是任意连续的三个数中，都一定要有一个比 3 大。满足要求的话就返回所有符合条件的数组，如果不存在符合条件的排列，就返回 -1

输入：5,3
输出：12345（任意的3个子串的数字都至少满足一个数比3大）
"""


def three_preson(n, x):
    d_x_cnt = n - x
    x_x_cnt = x
    if 2 * d_x_cnt < x_x_cnt:
        return -1

    # 构建数组
    x_x_arr = [i for i in range(1, x + 1)]
    d_x_arr = [i for i in range(x + 1, n + 1)]
    n1, n2 = len(x_x_arr), len(d_x_arr)
    rst_arr = []
    index = 0
    for i in range(0, n1, 2):
        if i + 1 <= n1 - 1:
            rst_arr.append(x_x_arr[i])
            rst_arr.append(x_x_arr[i + 1])
        else:
            rst_arr.append(x_x_arr[i])
        rst_arr.append(d_x_arr[i // 2])
        index = i
    if index + 1 < n2:
        for val in d_x_arr[i + 1 :]:
            rst_arr.append(val)
    return rst_arr


def main():
    T = int(input())
    for _ in range(T):
        n, x = map(int, input().split())
        ans = three_preson(n, x)
        print(ans)


# main()


# three_preson(3, 1)
t = three_preson(3, 6)
print(t)
