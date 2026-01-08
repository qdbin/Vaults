"""
    链接：https://leetcode.cn/problems/linked-list-cycle-ii/
"""
from typing import Optional

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        # 判断是否为空或就一个结点
        if not (head and head.next):
            return None

        # 将结点加入集合，若集合存在则直接返回集合中的元素
        seen=set()
        while head:
            if head in seen:
                return head
            else:
                seen.add(head)
                head=head.next
        return None