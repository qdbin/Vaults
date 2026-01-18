
"""
    链接：https://leetcode.cn/problems/longest-increasing-subsequence/solutions/2147040/jiao-ni-yi-bu-bu-si-kao-dpfu-o1-kong-jia-4zma
"""
from bisect import bisect_left
from typing import *

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        ng = 0  # g 的长度
        for x in nums:
            j = bisect_left(nums, x, 0, ng)
            nums[j] = x
            if j == ng:  # >=x 的 g[j] 不存在
                ng += 1 
        return ng


"""
    思想：动态规划dp[]
        - dfs记忆化搜索改dp[]数组,递归改循环
    1. 
"""
from typing import *

class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # dp[index]表示以对应下标结尾时的最长子串
        dp=[1]*len(nums)

        # 确定dp[i]的最长子串长度
        for i in range(len(nums)):
            # 每次确定dp[i]，都要遍历，判断前面的nums[index]是否存在比当前nums[i]小的，从而直接复用dp[index]的子串长度
            for j in range(i):
                if nums[j]<nums[i]:
                    dp[i]=max(dp[i],dp[j]+1)    # ! dp[j]+1
        return max(dp)
