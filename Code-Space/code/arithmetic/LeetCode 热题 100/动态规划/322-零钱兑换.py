"""
    链接：https://leetcode.cn/problems/coin-change/description/
    动态规划，自底向上
"""
from math import inf
from typing import List
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp=[float(inf)]*(amount+1)  # amount+1个！！！
        dp[0]=0     # 初始值！！！！！！！！

        for a in range(1,amount+1):
            for c in coins:
                if a-c>=0:
                    dp[a]=min(dp[a],dp[a-c]+1)  # dp[a-c]+1 !!!!!
        
        if dp[amount]==float(inf):
            return -1
        else:
            return dp[amount]