"""
    思想：双指针（循环收缩双指针）
    1. 循环收缩，直到双指针分别指向元音字符
    2. 交换双指针对应字符
    3. 然后返回
"""
class Solution:
    def reverseVowels(self, s: str) -> str:
        # Init（双指针，哈希Set）
        left,right=0,len(s)-1
        arr,st=list(s),set("aoeiuAOEIU")

        # 收缩双指针
        while left<right:
            # 确定left指针为元音
            while arr[left] not in st and left<right:
                left+=1
            # 确定right指针为元音
            while arr[right] not in st and right>left:
                right-=1
            
            # 交换
            arr[left],arr[right]=arr[right],arr[left]

            # 遍历收缩
            left,right=left+1,right-1
        
        return ''.join(arr)

if __name__=='__main__':
    cla=Solution()
    t=cla.reverseVowels("IceCreAm")
    print(t)