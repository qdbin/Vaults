"""
    思想：异或
"""
from typing import *

class Solution:
    def singleNumber(self, nums: List[int]) -> List[int]:
        # 1. 遍历 nums 执行异或运算
        x = 0
        for num in nums:  
            x ^= num 

        # 2. 返回出现一次的数字 x     
        return x;         