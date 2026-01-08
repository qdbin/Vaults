"""
    快慢双指针：快指针到底尾结点，慢指针刚好到底中间结点
"""
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def middleNode(self, head: ListNode) -> ListNode:
        # 快慢双指针
        fast = slow = head

        # 循环遍历
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        
        # 返回中间结点
        return slow