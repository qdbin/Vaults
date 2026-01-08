"""
    链接：https://leetcode.cn/problems/linked-list-cycle/description/
    
    思想：
        快慢指针，两者同时遍历结点，快指针如果与慢指针相等则说明有环，若快指针提前遍历到null，则说明无环

    notes：
        也可使用hashmap遍历存储每一个结点，如果字典中存在相同的结点，则说明有环
"""

from typing import Optional
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        # 筛选掉明head为空或没有下一个结点的情况
        if( not head or not head.next):
            return False
        
        # 第一次初始化，避免最开始导致fast==slow
        slow,fast=head.next,head.next.next
        # 快指针遍历速度为2，故此需要判断cur和next是否为none，如果为none则直接结束循环
        while fast and fast.next:
            if fast==slow:
                return True
            else:
                fast=fast.next.next
                slow=slow.next
        
        return False
            
        