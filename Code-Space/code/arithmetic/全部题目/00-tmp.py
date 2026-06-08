Mapping = ["", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]
from typing import List
from unicodedata import digit


class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        n, ans, path = len(digits), [], ['' * len(digits)]

        def dfs(i: int):
            if i == n:
                ans.append(''.join(path))
                return
            else:
                for c in Mapping[digit[i]]:
                    path[i] = c
                    dfs(i + 1)

            dfs(0)
            return ans
