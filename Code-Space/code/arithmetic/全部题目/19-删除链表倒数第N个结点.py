"""
    思想：双指针

    要点：
        1.在原有链表前加一个虚拟头结点，初始化left和right初始化位置，使right和left指针之间隔n个结点
        2.遍历right，当right指向none时，left刚好在倒数第n+1个结点，故此之间使用left.next=left.next.next即可实现结点删除

    实现：
        1.初始化left,h_dummy,使其均指向虚拟头结点，可理解此时left和h_dummy的下标均为0
        2.初始化right，从原有head结点遍历n次，可理解此时right下标为n
        3.然后left和right同步，直至right到底none结点，由于left和right之间隔了n个元素，此时left下标-(n+1)
        4.left.next=left.next即可删除结点，return h_dummy
"""

from typing import Optional
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:

        # 在原始head链表前加一个结点
        h_dummy=left=ListNode(0,head)
        right=head

        # 初始化right，使left和right之间隔了n个结点
        while n and right:  # 这里是right，不是right.next!!!!
            right=right.next
            n-=1
        
        # 同步遍历left、right，直至right遍历到链表的最后的none
        while right:
            left,right=left.next,right.next
        
        # 删除结点
        left.next=left.next.next

        # 返回原链表的第一个结点
        return h_dummy.next
        