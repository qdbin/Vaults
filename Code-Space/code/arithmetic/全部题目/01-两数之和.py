"""
    思想：通过哈希表实现

    实现：
        1.确定字典中是否已有target-nums[i]的数值，若有则返回两个数的下标，否则则添加当前值
"""
from typing import List
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dict={}
        for i in range(len(nums)):
            if target-nums[i] in dict:
                return (dict[target-nums[i]],i)
            else:
                dict[nums[i]]=i