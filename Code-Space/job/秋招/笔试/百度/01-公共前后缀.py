"""
    两个字符串得最少分（小明和小红，只能选两个字符串，小明确定字符串s，则小红只能选字符串t）
    1. 若 t="abcdxyz" s="xyzabcd",若小明选t，则小红选s，则得分为len("abcd")=4,反之得分为len("xyz")=3
"""
s=str(input())
t=str(input())

# s（前）t（后）
ans,ans1,ans2,len1,len2=0,0,0,len(s),len(t)
min_len=len1 if len1<len2 else len2

for i in range(min_len):
    if s[:i+1]==t[-(i+1):]:
        ans1=ans1+1
    else:
        continue

for i in range(min_len):
    if t[:i+1]==s[-(i+1):]:
        ans2=ans2+1
    else:
        continue

ans=ans1 if ans1<ans2 else ans2
print(ans2)