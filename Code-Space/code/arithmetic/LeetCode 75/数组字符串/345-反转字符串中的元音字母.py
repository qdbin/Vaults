"""
    思想：利用集合存储元音字母，双指针left<right遍历收缩，如果不是元音字母就收缩left、right直至left和right都是元音字母，然后交换字符，然后继续下一次循环

    注意：
        left>=right为结束条件，>=说明结束
"""
class Solution:
    def reverseVowels(self, s: str) -> str:
        l,r=0,len(s)-1
        arr=list(s)
        st=set('aeiouAEIOU')
        while l<r:
            while l<len(s) and arr[l] not in st:
                l+=1
            while r>=0 and arr[r] not in st:
                r-=1
            if l>=r:
                break
            else:
                arr[l],arr[r]=arr[r],arr[l]
                l+=1
                r-=1

        return ''.join(arr)

if __name__=='__main__':
    t='alklsdesldfi'
    print(Solution().reverseVowels(t))