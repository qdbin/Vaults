"""
    思想：栈迭代+颜色标记，while stack：不断弹出栈中的结点，并将其右左的结点加入到当前栈中，如果当前结点为none则遍历下一个栈

    链接：https://leetcode.cn/problems/binary-tree-inorder-traversal/solutions/25220/yan-se-biao-ji-fa-yi-chong-tong-yong-qie-jian-ming/
"""
from typing import Optional,List
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        white,gray=0,1
        res,stack=[],[]
        stack.append((white,root))
        while stack:
            color,node=stack.pop()
            if node is None:
                continue
            if color==white:
                stack.append((white,node.right))
                stack.append((gray,node))
                stack.append((white,node.left))
            else:
                res.append(node.val)
        return res


"""
    思想：递归遍历，流程：先左子树，再返回，再添加结点值，再右子树

    链接：https://leetcode.cn/problems/binary-tree-inorder-traversal/solutions/3738246/3bu-gao-ding-er-cha-shu-zhong-xu-bian-li-j20k/
"""

class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res=[]
        self.midOrder(res,root.left)
        return res
    
    def midOrder(self, res, root):
        # 2. 递归终止条件：遇到空节点则返回
        if root is None:
            return

        # 1.问题分解（递推公式）
        # 1.1 先左子树
        self.midOrder(res, root.left)
        # 1.2 再访问根节点
        res.append(root.val)
        # 1.3 最后去右子树
        self.midOrder(res, root.right)