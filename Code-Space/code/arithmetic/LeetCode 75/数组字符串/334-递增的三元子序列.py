"""
    思想：类似三指针，这里的one和two并不定死，先更新one，再更新two

    Example：
        [100,522,1,553,3533,2,3]   return true ( [1,2,3] )
    
"""

from math import inf
from typing import List
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        one,two=float (inf)

        for three in nums:
            # 若存在two，则说明总有比two更小的one，即若three>two,则说明前面又比其小的子序列，three>two>one
            if three>two:
                return True
            # 总是更新one的最小值
            elif three<one:
                one=three
            # 此处也在更新two的最小值（three比two小，更新two）
            else:
                two=three
