"""
  思想：和最长回文子串一个性质，中心拓展遍历所有字串，一个简单的改版而已！！！
"""
class Solution:
    def countSubstrings(self, s: str) -> int:
        ans=0
        for i in range(len(s)):
          # 奇拓展点
          l=r=i
          while l>=0 and r<len(s) and s[l]==s[r]:
              ans+=1
              l,r=l-1,r+1
              
          # 偶拓展点
          l,r=i,i+1
          while l>=0 and r<len(s) and s[l]==s[r]:
              ans+=1
              l,r=l-1,r+1
        
        return ans