"""
    思路：双指针
"""
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        # 双指针版本
        i, j = 0, 0        # 分别指向两个字符串的指针
        res = []
        while i < len(word1) or j < len(word2):  # 任一指针未到末尾
            if i < len(word1):
                res.append(word1[i])    # 添加word1的字符
                i += 1                  # word1指针前进
            if j < len(word2):
                res.append(word2[j])    # 添加word2的字符
                j += 1                  # word2指针前进
        return ''.join(res)             # 将字符列表连接成字符串