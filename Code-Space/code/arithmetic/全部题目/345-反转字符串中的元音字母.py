"""
    链接：https://leetcode.cn/problems/reverse-vowels-of-a-string/submissions/692326016/
    思想：双指针（循环收缩双指针）
    1. 循环收缩，直到双指针分别指向元音字符
    2. 交换双指针对应字符
    3. 然后返回
"""
class Solution:
    def reverseVowels(self, s: str) -> str:
        arr,seen=list(s),set("aeiouAEIOU")
        left,right=0,len(s)-1
        while left<right:
            if s[left] not in seen:
                left+=1
            elif s[right] not in seen:
                right-=1
            else:
                # 交换
                arr[left],arr[right]=arr[right],arr[left]
                #! 不要忘了i办理遍历收缩
                left,right=left+1,right-1
        return ''.join(arr)

if __name__=='__main__':
    cla=Solution()
    t=cla.reverseVowels("IceCreAm")
    print(t)