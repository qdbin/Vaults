"""
    链接：https://leetcode.cn/problems/group-anagrams/
    思想：哈希字典，利用defaultdice(list)实现
    1. 遍历strs,通过''.join(sorted(cur_str)),实现对str排序得key
    2. 相同的key,然后放入对应key的[]
"""
from collections import defaultdict
from typing import *
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # 如果 key 不在字典中，则自动插入空列表[]并返回，及默认val为[]
        dic=defaultdict(list)       # 工厂函数，本质是类，调用生成该类的示例

        # 遍历str,并放入对应key的[]
        for cur_str in strs:
            # 对当cur_str排序，作为dic的key，并利用dic[key]直接append添加cur_val
            sort_cur_str=''.join(sorted(cur_str))   # sorted()返回值是list，通过join连接
            # 相同sort_str即为相同key
            dic[sort_cur_str].append(cur_str)
        
        # 返回
        return list(dic.values())