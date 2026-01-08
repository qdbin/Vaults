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
    def addTwoNumbers(self,l1:Optional[ListNode],l2:Optional[ListNode])->Optional[ListNode]:
        list=ListNode()
        current_node=list
        jinwei=0
        x,y,jinwei,current_val=0,0,0,0
        while(l1 or l2):

            x=l1.val if l1 else 0
            y=l2.val if l2 else 0

            #获取“进位”和“个位”
            current_val=x+y+jinwei
            jinwei=current_val//10
            current_val=current_val%10

            #下一个结点
            current_node.next=ListNode(current_val) # 赋值next,并初始化val
            current_node=current_node.next
            if l1:l1=l1.next
            if l2:l2=l2.next
        if jinwei>0:
            current_node.next=ListNode(1)
        return list.next



if __name__ == '__main__':
    #定义结点
    anode1 = ListNode(1)
    anode2 = ListNode(2)
    anode3 = ListNode(3)
    bnode1 = ListNode(4)
    bnode2 = ListNode(5)
    bnode3 = ListNode(6)
    # 方法一：正序链接(使用原结点->node.next)
    bnode1.next = bnode2
    bnode2.next = bnode3
    # 方法二：倒序链接
    anode2.next = anode3
    anode1.next = anode2
    """
        # 方法二：正序链接(使用临时变量tmp->tmp.next)
        tmp=anode1
        tmp.next=anode2
        tmp=tmp.next
        tmp.next=anode3 
    """
    result=Solution().addTwoNumbers(anode1, bnode1)

    tmp = result
    while (tmp != None):
        x = tmp.val
        print(x)
        tmp = tmp.next

