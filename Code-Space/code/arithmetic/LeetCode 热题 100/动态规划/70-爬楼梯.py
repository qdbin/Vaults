"""
    题目：https://leetcode.cn/problems/climbing-stairs?envType=study-plan-v2&envId=top-100-liked
    DP动态规划全部存储
"""
class Solution:
    def climbStairs(self, n: int) -> int:
        pre,cur=1,2
        for _ in range(3,n+1):
            # 左边的cur和pre是当前遍历位置真实对应的的cur和pre，右边的则为上次遍历的cur和pre
            cur,pre=cur+pre,cur 
        return cur if n>=2 else (1 if n==1 else 0)


"""
    DP动态规划全部存储
"""
class Solution:
    def climbStairs(self, n: int) -> int:
        dp=[1]*n
        if n>=2:
            dp[1]=2
        for i in range(2,n):
            dp[i]=dp[i-1]+dp[i-2]
        
        return dp[n-1]