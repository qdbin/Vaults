"""
链接：https://leetcode.cn/problems/maximum-subarray/
"""

from typing import Optional, List


class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp = [0] * len(nums)
        dp[0] = nums[0]

        for i in range(1, len(nums)):
            if dp[i - 1] <= 0:
                dp[i] = nums[i]
            else:
                dp[i] = nums[i] + dp[i - 1]
        return max(dp)
