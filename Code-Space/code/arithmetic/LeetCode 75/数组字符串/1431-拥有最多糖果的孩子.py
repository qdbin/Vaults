"""
    思路：得到数组中最大的糖果值，tmp=最大糖果值-额外糖果，如果普遍都>tmp,则都为true
"""
from typing import *
class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        k = max(candies) - extraCandies
        return [c >= k for c in candies]