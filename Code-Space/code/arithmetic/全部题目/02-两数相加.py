"""
    链接：https://leetcode.cn/problems/add-two-numbers/description/
    # 定义result链表，同时也要赋值给l，便于遍历
    # while循环，链表结点要l1=l1.next遍历
    # 循环结束要判断进位是否为1，若为1要添加结点1
"""
# Definition for singly-linked list.
from typing import Optional


class ListNode:
    def __init__(self,val:int=0,next=None):
        self.val=val
        self.next=next

class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        head=ListNode()
        cur_node=head
        x,y,jinwei,cur_val=0,0,0,0
        while l1 or l2:
            x=0 if l1==None else l1.val
            y=0 if l2==None else l2.val
            
            # 获取“进位”和“个位”
            #! t=l1.val+l2.val+jinwei   ! 可能会遍历到当前结点为None的情况
            t= x+y+jinwei
            cur_val,jinwei = (t,0) if t<10 else (t%10,1)
            
            # 给下个结点赋值
            cur_node.next=ListNode(cur_val) # 给next赋值！！！不是给当前结点赋值！！！
            
            # 并统一遍历下个结点
            #! l1,l2=l1.next,l2.next    ! 错误同上
            cur_node=cur_node.next
            if l1:
                l1=l1.next
            if l2:
                l2=l2.next
            
        if jinwei:
            cur_node.next=ListNode(1)

        return head.next
    
if __name__=="__main__":
    l1=ListNode(2,ListNode(4,ListNode(3)))
    l2=ListNode(5,ListNode(6,ListNode(3)))
    t=Solution().addTwoNumbers(l1,l2)
    print(t)