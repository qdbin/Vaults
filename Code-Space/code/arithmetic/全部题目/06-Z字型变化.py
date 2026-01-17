"""
    链接：https://leetcode.cn/problems/zigzag-conversion/
    
    思路：创建一个指定行数的字符串列表，对应下标字符串存储指定行的字符
        1、顺序遍历字符串，安装横竖方向进行不断更新当前行，从而将遍历字符放到指定行的字符串中
"""
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        #! 特殊情况1一定要有！
        if numRows==1 or numRows>len(s):
            return s
        
        cur_row,going_down,row_arr=0,False,[""]*numRows
        for c in s:
            row_arr[cur_row]+=c
            
            # 到达第一行或最后一行，改变方向
            if cur_row==0 or cur_row==numRows-1:
                going_down=not going_down
            
            # 根据方向更新行
            cur_row+=1 if going_down else -1
        
        return "".join(row_arr)