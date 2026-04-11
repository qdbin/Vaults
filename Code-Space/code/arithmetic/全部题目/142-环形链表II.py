"""
链接：https://leetcode.cn/problems/linked-list-cycle-ii/
"""

# Definition for singly-linked list.
from typing import Optional


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return None

        slow = fast = head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
            # 有环则中断快指针！
            if slow == fast:
                break

        # 若快指针为空（无环）则返回
        if not fast or not fast.next:
            return None

        # 有环，将快指针恢复至起点，重写追赶！
        fast = head
        while fast != slow:  # !必须是fast!=slow直接判断，然后便于确认位置边界为0的特殊情况
            slow, fast = slow.next, fast.next
        return fast
