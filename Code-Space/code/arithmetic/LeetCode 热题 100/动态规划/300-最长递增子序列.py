


"""
    思想：DP动态规划存储（自底向上（由后向前））
    1. dp存储不同下标的子序列长度
    2. 由后向前确认子序列长度，每确认对应下标的子序列，都要遍历后面符合条件，从而获得不同子序列的dp,从而求出max_len
"""
from typing import *
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # 存储对应位置的子序列个数
        dp=[1]*len(nums)
        
        # 由后向前确定对应index的dp
        for i in reversed(range(len(nums))):

            # 判读对应下标，判读后面的子序列，是否满足大于当前的，并更新    # 这里不会越界，因为当i=n-1时，range(n,n-1)自动跳过
            for j in range(i+1,len(nums)):
                if nums[j]>nums[i]: 
                    dp[i]=max(dp[i],dp[j]+1)
        
        return max(dp)


"""
    思想：暴力递归+存储（自顶向下）（可用@cache）
    1. dfs(i)：表示以nums[i]开始的子序列长度
"""
from typing import List
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        def dfs(cur_index):
            # 查看字典，如果已被计算过则直接返回
            if cur_index in dic:
                return dic[cur_index]
                
            if cur_index==len(nums)-1:
                return 1
            
            max_len=1
            for i in range(cur_index+1,len(nums)):
                if nums[i]>nums[cur_index]:
                    max_len=max(max_len,dfs(nums,i)+1)
            
            # 存储
            dic[cur_index]=max_len  

            return max_len

        # 存储优化（存储指定下标的子序列长度）
        dic={}

        return max(dfs(i) for i in range(len(nums)))
    

"""
    思想：暴力递归（自顶向下）（dfs(i)，表示以nums[i]开始的子序列长度）
    1. 遍历原序列[]，传入下标计算不同下标所对应的dfs（子序列长度）
    2. dfs出口为最后一个下标，根据当前下标，递归剩余序列首个满足条件的
"""
from typing import List
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # dfs（返回子序列长度）（指定对应开始位置）
        def dfs(index):
            # 出口，当开始位置为最后一个，则无子序列
            if index==len(nums)-1:
                return 1

            # 初始化max_len，遍历剩余序列，找出合适的
            max_len=1
            for i in range(index+1,len(nums)):
                # 在剩余序列中找符合的，再递归计算
                if nums[i]>nums[index]:
                    max_len=max(max_len,dfs(nums,i)+1)
            
            return max_len

        # 遍历从不同位置开始计算子序列
        return max(dfs(i) for i in range(len(nums)))