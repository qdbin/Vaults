"""
    链接：https://leetcode.cn/problems/palindrome-number/description/
    推荐：字符串反转直接比较！及不推荐取模整除的half_x!(很慢)
    务必要先判断处x的个位数是否为0，如果为0直接false，不然如果模运算会有影响！！！
"""
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if 0<=x<10:
            return True
        elif x<0 or x%10==0:
            return False
        else:
            # 反转字符串
            if str(x)==str(x)[::-1]:
                return True
            else:
                return False
                