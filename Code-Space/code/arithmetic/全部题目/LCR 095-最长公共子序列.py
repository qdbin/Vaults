"""
    链接：https://leetcode.cn/problems/qJnOS7/solutions/2139882/jiao-ni-yi-bu-bu-si-kao-dong-tai-gui-hua-f5k9
    主要思想：两个字符对应的字符【选和不选】
"""

"""
    动态规划dp[]
"""
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # Init dp[][] 为避免下标越界，故此应保留dp[0][0],全部从dp[1][1]开始
        n,m=len(text1),len(text2)
        dp=[[0]*(m+1) for _ in range(n+1)]

        # 确定dp[i+1][j+1]  +1才表示对应的下标
        for i,x in enumerate(text1):
            for j,y in enumerate(text2):
                # 
                if x==y:
                    dp[i+1][j+1]=dp[i][j]+1
                else:
                    dp[i+1][j+1]=max(dp[i][j+1],dp[i+1][j])
        return dp[n][m]

"""
    暴力递归
"""
from functools import cache

class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        
        # 对应字符串下标（i，j）结尾的公共子串长度
        @cache
        def dfs(i,j):
            # 出口
            if i<0 or j<0:
                return 0
            
            # 相等情况，就直接返回dfs(i-1,j-1)
            if text1[i]==text2[j]:
                return dfs(i-1,j-1)+1
            else:
                return max(dfs(i-1,j),dfs(i,j-1))
        
        return dfs(len(text1)-1,len(text2)-1)