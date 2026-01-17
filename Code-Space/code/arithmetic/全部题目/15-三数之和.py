"""
    链接：https://leetcode.cn/problems/3sum/description/
    
    思想：排序+两数之和双指针
    实现：
        1.遍历排序后的列表，故以固定第一个值，再利用双指针的两数之和思想即可
    
    ！！！：
        由于要去重，故此排序后根遍历要确保 `nums[i]==nums[i-1]`且
"""
from typing import List
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:

        if not nums and len(nums)<3:
            return []

        nums.sort()     #! 排序！！！
        res=[]
        for i in range(len(nums)):
            # 确保下标>0,当前值与上一个值不同
            if i>0 and nums[i]==nums[i-1]:
                continue
            
            # 第一个值已确定，故此处转变为两数之和的双指针思想
            left,right=i+1,len(nums)-1
            while left<right:
                if(nums[i]+nums[left]+nums[right]==0):
                    res.append([nums[i],nums[left],nums[right]])
                    left+=1
                    #left<right 预防left下标越界
                    while nums[left]==nums[left-1] and left<right:
                        left+=1
                elif nums[i]+nums[left]+nums[right]<0:
                    left+=1
                elif nums[i]+nums[left]+nums[right]>0:
                    right-=1
            
        return res

if __name__ == '__main__':
    tmp=Solution().threeSum([0,0,0])
    print(tmp)