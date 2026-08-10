---
title: "1. 两数之和"
date: 2026-08-08
lastmod: 2026-08-08
draft: false
description: "记录两数之和的哈希表一次遍历写法：遍历当前数字时，在哈希表中查找 target - 当前值是否已经出现。"
summary: "两数之和可以用哈希表在一次遍历中完成：遍历当前数字时，查找 target - 当前值是否已经出现，若存在就返回两个下标。"
categories: ["LeetCode"]
tags: ["哈希表", "数组", "算法题"]
series: ["LeetCode"]
ShowToc: true
TocOpen: true
---

题目：[1. 两数之和 - 力扣（LeetCode）](https://leetcode.cn/problems/two-sum/description/?envType=study-plan-v2&envId=top-100-liked)

## 题目描述

给定一个整数数组 `nums` 和一个整数目标值 `target`，需要在数组中找出两个数，使它们的和等于 `target`。

返回这两个数在数组中的下标。题目保证每组输入只会对应一个答案，并且同一个元素不能使用两次。

## 核心思路

这道题可以用哈希表一次遍历解决。

假设当前遍历到的数字是 `value`，它的下标是 `key`。如果存在另一个数字 `b`，使得：

```text
value + b = target
```

那么：

```text
b = target - value
```

所以在遍历 `nums` 时，可以用一个哈希表 `record_map` 记录已经遍历过的数字和它们的下标。

每次遇到当前数字 `value`，先检查 `target - value` 是否已经在哈希表中：

- 如果存在，说明已经找到了答案；
- 如果不存在，就把当前数字和下标记录到哈希表中，继续向后遍历。

这样可以避免两层循环，把查找另一个数的过程从 `O(n)` 降到平均 `O(1)`。

## Python 实现

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        record_map = {}
        # a + b = c
        # c - a = b
        for key, value in enumerate(nums):
            if target - value in record_map:
                return [key, record_map[target - value]]
            record_map[value] = key

        return [-1, -1]
```

这里的 `record_map` 存储的是：

```text
数字 -> 下标
```

例如遍历到 `value = 7`、`target = 9` 时，只需要检查 `2` 是否已经出现在 `record_map` 中。如果出现过，就可以直接返回当前下标和 `2` 对应的下标。

## 为什么要先查再存

循环中要先判断 `target - value` 是否存在，再把当前 `value` 放进哈希表。

这样可以避免同一个元素被使用两次。

例如 `nums = [3]`、`target = 6` 时，如果先把 `3` 存入哈希表，再检查 `target - 3`，就可能错误地把同一个 `3` 当作两个数使用。

## 复杂度分析

- 时间复杂度：`O(n)`，只需要遍历数组一次；
- 空间复杂度：`O(n)`，最坏情况下哈希表会存储数组中的大部分元素。

## 容易出错的地方

- 哈希表中应该存下标，而不是只存数字；
- 不能用同一个元素两次，所以要先查找补数，再记录当前数字；
- 返回的是下标，不是两个数字本身；
- 有重复数字时，哈希表仍然可以正常处理，因为每次只查找已经遍历过的元素。
