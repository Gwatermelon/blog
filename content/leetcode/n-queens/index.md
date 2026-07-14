---
title: "51. N 皇后"
date: 2026-07-14
draft: false
description: "记录 N 皇后问题的经典回溯写法：按行放置皇后，检查列和两条对角线是否合法，并在回溯时恢复棋盘状态。"
categories: ["LeetCode"]
tags: ["回溯", "N 皇后", "算法题"]
series: ["LeetCode"]
---

题目：[51. N 皇后 - 力扣（LeetCode）](https://leetcode.cn/problems/n-queens/description/)

## 题目描述

按照国际象棋的规则，皇后可以攻击与之处在同一行、同一列或同一斜线上的棋子。

`n` 皇后问题研究的是：如何将 `n` 个皇后放置在 `n x n` 的棋盘上，并且使皇后彼此之间不能相互攻击。

给你一个整数 `n`，返回所有不同的 `n` 皇后问题的解决方案。

每一种解法包含一个不同的棋子放置方案，其中 `'Q'` 和 `'.'` 分别代表皇后和空位。

## 回溯思路

N 皇后是一道经典回溯题。核心思路是：按照行来放棋子。

也就是说，递归函数 `dfs(row)` 表示当前准备处理第 `row` 行。因为每一行只能放一个皇后，所以在这一行中枚举所有列，判断当前位置能不能放皇后：

- 如果当前位置合法，就放置皇后；
- 递归处理下一行；
- 递归结束后，把当前位置恢复成空位。

递归终止条件是 `row == n`。

以 `n = 4` 为例，当 `row == 4` 时，说明第 `0` 到第 `3` 行都已经处理完了，此时棋盘就是一个完整解法，可以加入答案。

## 如何判断当前位置是否合法

假设当前准备在 `(row, col)` 放皇后。因为我们是从上到下逐行放置，所以只需要检查已经放过皇后的区域：

- 同一列的上方是否有皇后；
- 左上对角线是否有皇后；
- 右上对角线是否有皇后。

文档里的实现还检查了当前行左侧。由于每一行在进入下一行前都会回溯恢复，当前行通常不会残留皇后，这个检查不是必须的，但保留它也不影响正确性。

## Python 实现

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        chess = [['.' for _ in range(n)] for _ in range(n)]
        ans = []

        def dfs(row, chess):
            if row == n:
                ans.append([''.join(r) for r in chess])
                return

            for i in range(n):
                if is_valid(row, i, chess):
                    chess[row][i] = 'Q'
                    dfs(row + 1, chess)
                    chess[row][i] = '.'

        def is_valid(row, col, chess):
            if 0 <= row <= n - 1 and 0 <= col <= n - 1:
                x = row - 1
                while x >= 0:
                    if chess[x][col] == 'Q':
                        return False
                    x -= 1

                y = col - 1
                while y >= 0:
                    if chess[row][y] == 'Q':
                        return False
                    y -= 1

                x, y = row - 1, col - 1
                while x >= 0 and y >= 0:
                    if chess[x][y] == 'Q':
                        return False
                    x -= 1
                    y -= 1

                x, y = row - 1, col + 1
                while x >= 0 and y <= n - 1:
                    if chess[x][y] == 'Q':
                        return False
                    x -= 1
                    y += 1

                return True

            return False

        dfs(0, chess)
        return ans
```

## 关键点

这道题的关键不是一次性把所有皇后放好，而是把问题拆成一行一行处理。

递归过程中的状态是当前棋盘 `chess` 和正在处理的行号 `row`。每次选择一个合法列放置皇后，然后进入下一层递归。如果后续走不通，就撤销当前选择，继续尝试这一行的下一个列。

这就是典型的回溯结构：

```python
做选择
递归
撤销选择
```

## 复杂度分析

- 时间复杂度：近似 `O(n!)`。每一行都要选择一个列，并且后续可选位置会逐渐减少。
- 空间复杂度：`O(n^2)`，主要来自棋盘存储；递归栈深度为 `O(n)`。

## 容易出错的地方

- `row == n` 时要立刻收集答案并返回；
- 加入答案时要把每一行转换成字符串，不能直接把二维列表引用放进答案；
- 回溯后必须把 `chess[row][i]` 恢复成 `'.'`；
- 对角线检查时，左上和右上的坐标更新方向不要写反。
