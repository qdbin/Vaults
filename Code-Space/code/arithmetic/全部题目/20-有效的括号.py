"""
    链接：https://leetcode.cn/problems/valid-parentheses/description/
    思想：栈进左括号，右括号与栈顶匹配

    实现细节：
        1.利用字典存储匹配的括号，如：dic={'(':')','[':']','{':'}'}
        2.利用列表arr，append进栈，pop出栈，arr[-1]表示栈顶元素
        3.利用最后对栈中的元素个数进行判断，如果为空则说明全部为有效括号

    注意细节：
        1.arr[-1]拿栈顶元素时，要确保栈列表不为空！！！
"""
class Solution:
    def isValid(self, s: str) -> bool:
        dic={'(':')','[':']','{':'}'}
        stack=[]

        for char in s:
            # 如果char在字典中，即char=='([{'
            if char in dic:
                stack.append(char)
            # 反之，char=='}])'
            else:
                # 如果栈不为空，且char于栈顶的'([{'的字典的右括号匹配，则弹出栈
                if stack and char == dic[stack[-1]]:
                    stack.pop()
                else:
                    return False

        # 判断最后的栈是否为空，是则返回true
        if len(stack)==0:
            return True
        else:
            return False

if __name__=='__main__':
    t=Solution().isValid('()')