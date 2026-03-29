"""
这个题目是给定一个数组，再给定一个查询值k。数组就是给你一个整形的数组，然后再给一个查询k。然后去返回这个数组里面的所有子串，所有子串满足和小于k的个数。

对，比如说12345，对不对？然后它里面的所有的子串，我给了一个k，然后求所有的子串的个数。然后可以动态规划做
"""


def main(arr, k):
    """动态规划,dp[l][r]表示下班区别的和"""
    # print(f'k:{k}')
    dp = [[0] * len(arr) for _ in range(len(arr))]
    ans = 0
    for i in range(len(arr)):
        dp[i][i] = arr[i]
        if arr[i] <= k:
            ans += 1
    # print(ans)
    if ans == 0:
        return 0
    for l in range(len(arr)):
        for r in range(l + 1, len(arr)):
            if arr[r] + dp[l][r - 1] <= k:
                ans += 1
                dp[l][r] = arr[r] + dp[l][r - 1]
                # print(l,r)
            else:
                break
    return ans


def start():
    n, q = map(int, input().split())
    arr = list(map(int, input().split()))
    for _ in range(q):
        k = int(input())
        ans = main(arr, k)
        print(ans)
        # print('\n')


start()
# ans=main([1,5,6],10)
# print(ans)
