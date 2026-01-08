"""
    思想：链表中点+反转链表+对比
    1. 之所以要获得中点是因为反转链表只能反转一半，不能全部反转，否则开销空间必然会吃不消
"""

from copy import deepcopy
from typing import *

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    # 获得链表中点
    def middleNode(self,head)->ListNode:
        slow=fast=head
        while fast and fast.next:
            slow,fast=slow.next,fast.next.next
        return slow

    # 反转链表
    def reversalLink(self,head)->ListNode:
        # cur,pre=deepcopy(head),None   # 深拷贝，可以不用获得中点
        cur,pre=head,None
        while cur:
            cur.next,pre,cur=pre,cur,cur.next
        return pre
    
    # 对比判断是否回文
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        minddle_node=self.middleNode(head)
        head2=self.reversalLink(minddle_node)

        while head2:
            if head.val!=head2.val:
                return False
            head,head2=head.next,head2.next
        return True

if __name__=='__main__':
    cla=Solution()
    head=ListNode(1)
    head=ListNode(2,head)
    head=ListNode(2,head)
    head=ListNode(1,head)
    print(cla.isPalindrome(head))
