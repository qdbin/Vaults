"""
    背包道具得分（不同道具对应不同得分）
    如道具得分为：1、3、4、5、6 若拿得分为4的道具，拿得分为3的道具将销毁，故计算在道具队列得分最多该如何得分
"""

num=int(input())
arr=list(map(int,input().split()))

def max_score(n,values):
    # 统计分值分值
    count=[0]*100001
    for value in values:
        count[value]+=1
    
    # 动态规划遍历赋值
    dp=[0]*100001
    dp[1]=count[1]*1

    for i in range(2,100001):
        dp[i]=max(dp[i-1],dp[i-2]+count[i]*i)

    return dp[100000]

ans=max_score(num,arr)
print(ans)