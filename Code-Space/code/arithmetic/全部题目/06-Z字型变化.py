"""
链接：https://leetcode.cn/problems/zigzag-conversion/

思路：创建一个指定行数的字符串列表，对应下标字符串存储指定行的字符
    1、顺序遍历字符串，安装横竖方向进行不断更新当前行，从而将遍历字符放到指定行的字符串中
"""


class Solution:
    def convert(self, s: str, numRows: int) -> str:
        #! 特殊情况1一定要有！
        if numRows == 1 or len(s) < numRows:
            return s

        rows = [""] * numRows
        cur_row, is_down = 0, True

        for c in s:
            rows[cur_row] += c
            if is_down:
                cur_row += 1
            else:
                cur_row -= 1
            if cur_row == 0 or cur_row == numRows - 1:
                is_down = not is_down
        return "".join(rows)
