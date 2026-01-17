from typing import *
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        if not nums and len(nums)<3:
            return []
        nums.sort()
        ans=[]
        for i in range(len(nums)):
            if i!=0 and nums[i]==nums[i-1]:
                continue
            
            l,r=i+1,len(nums)-1
            while l<r:
                target=-nums[i]-nums[l]
                if target==nums[r]:
                    ans.append([nums[i],nums[l],nums[r]])
                    l+=1
                    while nums[l]==nums[l-1] and l<r:
                        l+=1
                elif target<nums[r]:
                    r-=1
                else:
                    l+=1
        
        return ans