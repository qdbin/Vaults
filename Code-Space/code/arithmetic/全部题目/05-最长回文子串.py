"""
    链接：https://leetcode.cn/problems/longest-palindromic-substring/
    # 遍历中心点[l,r]（奇回文的中心点是：[0,0],[1,1],[2,2],[3,3]……[n-1,n-1],偶回文的中心点是：[0,1],[1,2],[2,3]……[n-2,n-1]）
    # 依次判断s[l]==s[r],若相同则继续向外拓展循环
    中心拓展法：https://leetcode.cn/problems/longest-palindromic-substring/solutions/2958179/mo-ban-on-manacher-suan-fa-pythonjavacgo-t6cx/
"""

class Solution:
    def longestPalindrome(self,s:str)->str:
        ans_l,ans_r=0,0     # l,r:子串的下标区间；ans_l,ans_r:最终回文串的下标区间

        # 奇回文判断（i充当中心点，向外拓展判断是否为回文）
        for i in range(len(s)):
            l=r=i
            while(l>=0 and r<len(s) and s[l]==s[r]):    #! s[l]==s[r]放后面！避免先出现下标越界再判断的情况
                l,r=l-1,r+1

            # 恢复while最后一次循环的影响
            l,r=l+1,r-1     
            # 判断是否比当前结果下标长度长，如果是则更新结果下标
            if (r-l+1)>(ans_r-ans_l):
                ans_l,ans_r=l,r

        # 偶回文判断（i,i+1充当中心点，向外拓展判断是否为回文）
        for i in range(len(s)-1):
            l,r=i,i+1
            while(l>=0 and r<len(s) and s[l]==s[r]):
                l,r=l-1,r+1

            l,r=l+1,r-1
            if (r-l+1)>(ans_r-ans_l):
                ans_l,ans_r=l,r
        
        # 此次由于是左闭右开，故右区间+1
        return s[ans_l:ans_r+1] 


        