"""
    链接：https://leetcode.cn/problems/XltzEq
    思想：收缩双指针，跳过非字母和数字，然后按原有逻辑验证
"""
class Solution:
    def isPalindrome(self, s):
        left, right, flag = 0, len(s) - 1, False
        # 收缩左右指针，都符合再对比，遍历
        while left <= right:
            if not s[left].isalnum():
                left += 1
            elif not s[right].isalnum() and right > left:
                right -= 1
            else:
                if s[left].lower() != s[right].lower():
                    return False
                left += 1
                right -= 1
        return True