"""
链接：https://leetcode.cn/problems/XltzEq
思想：收缩双指针，跳过非字母和数字，然后按原有逻辑验证
"""


class Solution:
    def isPalindrome(self, s: str) -> bool:
        # !lower()别忘了，以及右侧赋值
        s = s.lower()
        l, r = 0, len(s) - 1
        while l <= r:
            if s[l].isalnum() and s[r].isalnum():
                if s[l] != s[r]:
                    return False
                l, r = l + 1, r - 1
            elif not s[l].isalnum():
                l += 1
            else:
                r -= 1
        return True
