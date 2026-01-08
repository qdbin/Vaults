"""
    思想：双指针迭代，pre为反转的结点，cur为待反转的结点（时刻更新）
    1. 常规赋值链表也是从底层开始的
"""
from typing import *

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur,pre=head,None

        # 遍历反转结点
        while cur:
            # 平行赋值语法，等价方案
            cur.next,pre,cur=pre,cur,cur.next    # 有顺序！！！

            """
            tmp=cur.next    # 暂存next_node
            cur.next=pre    # 设置下个结点为pre_node
            pre=cur         # 更新pre_node=cur_node
            cur=tmp         # cur_node更新为下个待转换的结点
            """

        # 返回pre,即开端
        return pre