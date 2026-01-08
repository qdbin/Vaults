from typing import List

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        """
        计算二维网格中岛屿的数量
        
        @param {List[List[str]]} grid - 二维字符网格，'1'表示陆地，'0'表示水域
        @returns {int} 岛屿的总数量
        """
        n1, n2 = len(grid), len(grid[0])  # 获取网格的行数和列数
        cnt = 0  # 岛屿计数器

        def dfs(x: int, y: int) -> None:
            """
            深度优先搜索函数，用于标记连通的陆地，x,y为对应坐标
            """
            # 边界检查：越界或者不是陆地('1')则返回
            # ! grid[x][y] != '1' or x >= n1 or y >= n2
            if x < 0 or y < 0 or x >= n1 or y >= n2 or grid[x][y] != '1':
                return
            
            # 将当前陆地标记为已访问（用'2'表示）
            grid[x][y] = '2'
            
            # 递归搜索四个方向的相邻位置
            dfs(x - 1, y)  # 上
            dfs(x + 1, y)  # 下
            dfs(x, y - 1)  # 左
            dfs(x, y + 1)  # 右
        
        # 遍历整个网格
        for i in range(n1):
            for j in range(n2):
                # 如果发现未访问的陆地，说明找到了一个新岛屿
                if grid[i][j] == '1':
                    dfs(i, j)  # 使用DFS标记整个岛屿
                    cnt += 1   # 岛屿计数加1
  
        return cnt


if __name__ == '__main__':
    # 测试用例：应该返回1（一个大岛屿）
    test_grid = [
        ["1", "1", "1", "1", "0"],
        ["1", "1", "0", "1", "0"], 
        ["1", "1", "0", "0", "0"],
        ["0", "0", "0", "0", "0"]
    ]
    
    result = Solution().numIslands(test_grid)
    print(f"岛屿数量: {result}")
    
    # 额外测试用例：多个岛屿
    test_grid2 = [
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],
        ["0", "0", "0", "1", "1"]
    ]
    
    result2 = Solution().numIslands(test_grid2)
    print(f"第二个测试用例岛屿数量: {result2}")  # 应该返回3