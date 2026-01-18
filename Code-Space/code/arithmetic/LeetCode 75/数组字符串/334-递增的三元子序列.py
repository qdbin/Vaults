"""
    链接：https://leetcode.cn/problems/increasing-triplet-subsequence/
    思想：类似三指针，这里的one和two并不定死，先更新one，再更新two

    Example：
        [100,522,1,553,3533,2,3]   return true ( [1,2,3] )
    
"""

from math import inf
from typing import List
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        one,two=float(inf)
        
        for three in nums:
            # 总是更新one的最小值
            if three<one:
                one=three
            else:
                # 其次更新two
                if three<two:
                    two=three
                else:
                    return True
            