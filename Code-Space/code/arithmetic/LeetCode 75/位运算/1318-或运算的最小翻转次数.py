class Solution:
    def minFlips(self, a: int, b: int, c: int) -> int:
        cnt = 0
        while a or b or c:
            # 获取a，b，c的当前位
            a_bit = a & 1
            b_bit = b & 1
            c_bit = c & 1
            # 如果c的当前位为0，则a和b的当前位都必须为0，因此每有一个1就要翻转1次
            # 如果c的当前位为1，则a和b的当前位至少有一位为1，即如果都为0则要翻转1次
            cnt += (a_bit | b_bit == 0) if c_bit else (a_bit + b_bit)
            # 右移a,b,c更新最低位
            a >>= 1
            b >>= 1
            c >>= 1
        return cnt