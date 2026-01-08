from typing import List
class Solution:

    def canPlaceFlowers_01(self, flowerbed: List[int], n: int) -> bool:
        i, m = 0, len(flowerbed)
        while i < m:
            if (i == 0 or flowerbed[i - 1] == 0) and flowerbed[i] == 0 and (i == m - 1 or flowerbed[i + 1] == 0):
                n -= 1
                i += 2  # 下一个位置肯定不能种花，直接跳过
            else:
                i += 1
        return n <= 0

    def canPlaceFlowers_02(self, flowerbed: List[int], n: int) -> bool:
        if not n:
            return True

        for i in range(len(flowerbed)):
            if flowerbed[i]:
                continue
            else:
                if i==0 and flowerbed[i]==0 and flowerbed[i+1]==0:
                    flowerbed[i]=1
                    n-=1
                    
                if i!=0 and i!=len(flowerbed)-1 and flowerbed[i]==0 and flowerbed[i-1]==0 and flowerbed[i+1]==0:
                    flowerbed[i]=1
                    n-=1

                if i==len(flowerbed)-1 and flowerbed[i]==0 and flowerbed[i-1]==0:
                    flowerbed[i]=1
                    n-=1

            if n==0:
                break
        
        if not n:
            return True
        else:
            return False

if __name__=='__main__':
    t=Solution().canPlaceFlowers([1,0,0,0,1,0,0],2)