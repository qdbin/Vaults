"""
    链接：https://leetcode.cn/problems/reverse-integer/
    
    # int强转str，将str倒序重新赋值为str，str强转int自动去除0
"""
class Solution:
    def reverse(self, x: int) -> int:
        #int强转str
        str_x=str(x)
        if(str_x[0]=='-'):
            #去除‘-’：左开右闭，字符串倒序
            str_x=str_x[:0:-1]  #! 此次是[:::],而非[:,]
            #str强转int，自动去除开头的0
            x=int(str_x)
            x=-x
        else:
            #反转字符串
            str_x=str_x[::-1]
            #强转int，去除开头的0
            x=int(str_x)
        
        # 这里要判断是最终结果
        return x if -2147483648<=x<=2147483647 else 0