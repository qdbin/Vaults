"""
    思想：排序+双指针
    
    实现：
        1.通过enumerate，实现arr[(value,index)],排序arr通过left、right下标和01取value和index
        2.left+right==target则返回，若小则left++，若大则right++
"""
from typing import List
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:

        arr=[(value,index) for index,value in enumerate(nums)]
        arr.sort()

        left,right=0,len(nums)-1
        while True:
            if target==arr[left][0]+arr[right][0]:
                return (arr[left][1],arr[right][1])
            elif target<arr[left][0]+arr[right][0]:
                right=right-1
            elif target>arr[left][0]+arr[right][0]:
                left=left+1
