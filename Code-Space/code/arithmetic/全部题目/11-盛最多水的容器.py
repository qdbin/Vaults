"""
链接：https://leetcode.cn/problems/container-with-most-water/description/
# 双指针：用l,r代表链表的左右边界的下标，while l<r 循环,if else切换边界向内移动依次缩减
# 图示：https://leetcode.cn/problems/container-with-most-water/solutions/11491/container-with-most-water-shuang-zhi-zhen-fa-yi-do/
"""

from typing import List


class Solution:
    def maxArea(self, height: List[int]) -> int:
        l, r, res = 0, len(height) - 1, 0
        while l < r:
            if height[l] <= height[r]:
                # 计算面积，最大值重新赋值res
                res = max(res, height[l] * (r - l))
                # 向内缩进
                l += 1
            else:
                res = max(res, height[r] * (r - l))
                r -= 1
        return res
