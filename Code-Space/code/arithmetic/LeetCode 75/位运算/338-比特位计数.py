from typing import List
class Solution:
    def countBits(self, n: int) -> List[int]:
        # 枚举数字x的每一个二进制位，取为1的个数
        def bits(x: int) -> int:
            cnt = 0
            for i in range(32):
                if (x >> i) & 1:
                    cnt += 1
            return cnt

        # 枚举[0,n]的所有数字x
        return [bits(x) for x in range(n+1)]
