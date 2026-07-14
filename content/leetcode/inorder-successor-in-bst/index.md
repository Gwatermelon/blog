---
title: "LCR 053. 二叉搜索树中的中序后继"
date: 2026-07-14
draft: false
description: "记录二叉搜索树中序后继问题的三种写法：中序遍历记录节点、遍历全树维护候选答案，以及利用 BST 性质迭代查找。"
categories: ["LeetCode"]
tags: ["二叉搜索树", "中序遍历", "算法题"]
series: ["LeetCode"]
---

题目：LCR 053. 二叉搜索树中的中序后继 - 力扣（LeetCode）

## 这道题和中序遍历的关系

在二叉搜索树中，节点 `p` 的中序后继等价于：

> 所有值大于 `p.val` 的节点中，值最小的那个节点。

例如下面这棵二叉搜索树：

```text
        5
       / \
      3   8
     / \ / \
    2  4 6  9
```

它的中序遍历顺序是：

```text
2, 3, 4, 5, 6, 8, 9
```

如果 `p = 5`：

- 中序遍历中，`5` 的下一个节点是 `6`；
- 所有比 `5` 大的节点是 `6、8、9`；
- 其中最小的是 `6`。

根据这个逻辑，可以先使用中序遍历记录节点顺序，然后返回 `p` 所在位置的下一个节点。

## 方法一：中序遍历记录节点列表

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        find_list = []

        def dfs(node):
            if not node:
                return

            dfs(node.left)
            find_list.append(node)
            dfs(node.right)

        dfs(root)

        if p in find_list:
            index = find_list.index(p)
            if index != len(find_list) - 1:
                return find_list[index + 1]

        return None
```

这个方法直接利用了二叉搜索树的中序遍历结果有序这一点。缺点是需要额外的列表维护遍历顺序。

## 方法二：遍历整棵树维护候选答案

也可以遍历整棵树，并在遍历过程中维护一个变量 `ans`，用于记录当前已经找到的、大于 `p.val` 的最小节点。

这种写法不需要用列表维护顺序。因为这里遍历的是整棵树，所以前序、中序、后序都可以。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        ans = None

        def dfs(node):
            if not node:
                return

            nonlocal ans
            if node.val > p.val:
                if not ans or ans.val > node.val:
                    ans = node

            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return ans
```

## 方法三：利用二叉搜索树性质

上面的做法都没有充分利用二叉搜索树左小右大的性质。

可以通过比较当前节点和目标节点 `p` 的大小来缩小搜索范围：

- 如果 `root.val > p.val`，说明当前节点是一个可能的后继节点。先记录它，然后继续去左子树找更小但仍然大于 `p.val` 的节点。
- 如果 `root.val <= p.val`，说明当前节点以及它的左子树都不可能是答案，应该去右子树查找。
- 如果没有找到符合要求的节点，`ans` 会保持为 `None`。

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode':
        ans = None

        if not root:
            return None

        while root is not None:
            if root.val > p.val:
                ans = root
                root = root.left
            else:
                root = root.right

        return ans
```

这个方法利用了二叉搜索树的有序性，不需要完整遍历所有节点，也不需要额外列表。
