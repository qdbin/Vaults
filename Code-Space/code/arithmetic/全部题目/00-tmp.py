import copy
from math import inf
from typing import *


class ListNode:
    def __init__(self, val=0, next=None) -> None:
        self.val = val
        self.next = next


class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        def fanzhuan(head):
            cur, pre = head, None
            while cur:
                cur.next, pre, cur = pre, cur, cur.next
            return pre
