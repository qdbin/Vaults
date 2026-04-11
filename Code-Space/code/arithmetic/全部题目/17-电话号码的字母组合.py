Mapping = ["", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]
from typing import List


class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        n, ans, path = len(digits), [], [''] * len(digits)

        def dfs(i: int) -> None:
            if i == n:  # !出口
                ans.append(''.join(path))
                return
            else:
                for c in Mapping[digits[i]]:
                    path[i] = c
                    dfs(i + 1)

        dfs(0)
        return ans
