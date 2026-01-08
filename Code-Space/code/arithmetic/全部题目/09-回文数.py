"""
    务必要先判断处x的个位数是否为0，如果为0直接false，不然如果模运算会有影响！！！
"""
class Solution:
    def isPalindrome(self,x: int) -> bool:
        # 若x是负数 or 个位数为0，直接false
        if(x<0 or (x%10==0 and x!=0)):
            return False
            
        # 若x为[0~10)直接true
        elif(x>=0 and x<10):
            return True
        else:
            s=str(x)
            left,right=0,len(s)-1
            while(left<right):
                if s[left]!=s[right]:
                    return False
                left=left+1
                right=right-1
            return True
                