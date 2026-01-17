"""
    "https://leetcode.cn/problems/median-of-two-sorted-arrays/description/"
    思想：
        通过固定绳子长度，通过二分确定nums1列表的分割点i，来获得nums2列表的分割点j
    题解：
        确定绳子长度：(len(nums1)+len(nums2)+1)//2  确保绳子长度比余下的长度<=1
        确定nums1分割点i：通过二分确定，直至
        确定nums2分割点j：
        
    描述：
        给定两个大小分别为 m 和 n 的正序数组 nums1 和 nums2。找出并返回这两个正序数组的 中位数 。时间复杂度应该为 O(log (m+n)) 

    注意事项（250909）：
        ！！！：len(nums1)<=len(nums2),便于快速二分确定分割点i
        ！！！：如果交换nums列表，一定要注意对应的列表长度是否交换（可能存在先求出列表长度，但交换列表后并未交换列表长度）
"""
from math import inf
from typing import List

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # 前小后大！！！！
        if(len(nums1)>len(nums2)):
            nums1,nums2=nums2,nums1
        n1,n2=len(nums1),len(nums2)
        n=n1+n2
        
        # 此处的right主要用于确定"分割点"，即说明nums1左侧0或n个元素
        left,right=0,n1     # 此处是n1!!!!!!!!!!!!
        while(left<=right): #! 此处是等于！！！（这里是确定中点的）

            i=(left+right)//2       # 这里是left+right!!!!!!不是减是加，是加，老错！！！
            j=(n+1)//2-i            # 这里是n+1 !!!

            il=float(-inf) if i==0 else nums1[i-1]
            ir=float(inf) if i==n1 else nums1[i]
            jl=float(-inf) if j==0 else nums2[j-1]
            jr=float(inf) if j==n2 else nums2[j]

            if(il<=jr and jl<=ir):
                if n&1:
                    return max(il,jl)
                else:
                    return (max(il,jl)+min(ir,jr))/2

            if(il>jr):
                right=i-1

            elif(jl>ir):
                left=i+1





if __name__=='__main__':
    t=Solution().findMedianSortedArrays([1,3],[2])
    print(t)
