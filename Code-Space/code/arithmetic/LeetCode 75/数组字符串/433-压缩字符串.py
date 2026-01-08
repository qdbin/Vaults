from typing import List

class Solution:
    def compress(self, chars: List[str]) -> int:
        i,j,res= 0,0,[]
        while i < len(chars):
            cur,cnt = chars[i],1

            while i+1 < len(chars) and chars[i+1] == cur:
                cnt += 1
                i += 1

            chars[j],j=cur,j+1

            # 将对应位置的字符数组进行修改
            if cnt > 1:
                k = j 
                while cnt:
                    chars[j] = str(cnt % 10)
                    cnt //= 10 
                    j += 1
                x, y = k, j-1
                while x < y:
                    chars[x], chars[y] = chars[y], chars[x] 
                    x += 1 
                    y -= 1
            i += 1
        return j