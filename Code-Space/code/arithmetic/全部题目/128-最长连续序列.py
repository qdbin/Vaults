"""
    https://leetcode.cn/problems/longest-consecutive-sequence/description/
    思想：利用哈希集合，遍历set，通过val-1不在set，从而直接确定val为【开端】
"""
from typing import *


class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        # list转set
        s,ans=set(nums),0
        
        # 遍历set,确定val为【开端】，再确定子序列的长度
        for val in s:
            # 通过val-1不在set，从而确定val为set的【开端】
            if val-1 not in s:
                x=val
                while x in s:
                    x=x+1
                x=x-1
                ans=max(ans,x-val+1)

                # 优化
                if ans*2>=len(s):
                    return ans            
            else:
                continue

        return ans
