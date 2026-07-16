---
title: "146. LRU 缓存"
date: 2026-07-14
lastmod: 2026-07-14
draft: false
description: "记录 LRU 缓存的经典实现：哈希表负责 O(1) 查找，双向链表负责维护最近使用顺序。"
summary: "用哈希表和双向链表实现 O(1) 的查询、更新与淘汰，并梳理哨兵节点和边界处理。"
categories: ["LeetCode"]
tags: ["哈希表", "双向链表", "LRU", "算法题"]
ShowToc: true
TocOpen: true
series: ["LeetCode"]
---

题目：[146. LRU 缓存 - 力扣（LeetCode）](https://leetcode.cn/problems/lru-cache/)

LRU 是很经典也很重要的面试题，一定要熟练掌握。

## 题目描述

请你设计并实现一个满足 LRU（最近最少使用）缓存约束的数据结构。

实现 `LRUCache` 类：

- `LRUCache(int capacity)`：以正整数 `capacity` 初始化 LRU 缓存；
- `int get(int key)`：如果关键字 `key` 存在于缓存中，则返回关键字的值，否则返回 `-1`；
- `void put(int key, int value)`：如果关键字 `key` 已经存在，则变更其数据值 `value`；如果不存在，则插入该组 `key-value`；
- 如果插入操作导致关键字数量超过 `capacity`，则应该逐出最久未使用的关键字。

题目要求 `get` 和 `put` 都必须以 `O(1)` 的平均时间复杂度运行。

## 核心思路

这道题主要考察两个结构的配合：

- 哈希表：通过 `key` 在 `O(1)` 时间内找到节点；
- 双向链表：维护节点的新旧顺序，并支持在 `O(1)` 时间内删除和插入节点。

我们可以设计一个 `dummy` 哨兵节点，让它同时作为链表的虚拟头尾：

- `dummy.next` 指向最新使用的节点；
- `dummy.prv` 指向最久未使用的节点；
- 初始时，`dummy.next = dummy`，`dummy.prv = dummy`。

每次访问或更新一个节点时，都把它移动到链表头部。这样链表尾部的节点就是最久未使用的节点，容量超出时直接删除 `dummy.prv` 即可。

## 两个核心操作

为了维护双向链表，核心是写好两个函数：

- `remove(node)`：从链表中移除当前节点；
- `move_to_head(node)`：把节点加入链表头部，也就是放到 `dummy` 后面。

`remove` 的关键是跳过当前节点：

```python
def remove(self, node):
    node.prv.next = node.next
    node.next.prv = node.prv
```

`move_to_head` 的作用是把节点插入 `dummy` 和原头节点之间：

```text
dummy <-> original_head

插入 new_head 后：

dummy <-> new_head <-> original_head
```

对应四步操作：

1. 让 `node.next` 指向原头节点；
2. 让 `node.prv` 指向 `dummy`；
3. 让原头节点的 `prv` 指向 `node`；
4. 让 `dummy.next` 指向 `node`。

这里要注意第 3 步要在第 4 步之前执行。否则如果先改了 `dummy.next`，就拿不到原来的头节点了。

## Python 实现

```python
class Node:
    __slots__ = ('prv', 'next', 'key', 'value')

    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prv = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.dummy = Node(0, 0)
        self.dummy.prv = self.dummy
        self.dummy.next = self.dummy
        self.key_to_node = {}
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key in self.key_to_node:
            find_node = self.key_to_node[key]
            self.remove(find_node)
            self.move_to_head(find_node)
            return find_node.value

        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.key_to_node:
            find_node = self.key_to_node[key]
            find_node.value = value
            self.remove(find_node)
            self.move_to_head(find_node)
            return

        new_node = Node(key, value)
        self.key_to_node[key] = new_node
        self.move_to_head(new_node)

        if len(self.key_to_node) > self.capacity:
            old_node = self.dummy.prv
            del self.key_to_node[old_node.key]
            self.remove(old_node)

    def remove(self, node):
        node.prv.next = node.next
        node.next.prv = node.prv

    def move_to_head(self, node):
        node.next = self.dummy.next
        node.prv = self.dummy
        self.dummy.next.prv = node
        self.dummy.next = node
```

## 为什么是 O(1)

`get` 操作中：

- 用哈希表根据 `key` 找节点是 `O(1)`；
- 从双向链表中删除节点是 `O(1)`；
- 把节点移动到头部是 `O(1)`。

`put` 操作中：

- 如果 `key` 已存在，更新值并移动到头部，整体是 `O(1)`；
- 如果 `key` 不存在，新建节点并插入头部，整体是 `O(1)`；
- 如果容量超出，删除尾部节点也是 `O(1)`。

因此两个接口都满足题目要求。

## 容易出错的地方

- `get` 命中后也要把节点移动到头部，因为它刚刚被使用过；
- `put` 更新已有 `key` 时，也要把节点移动到头部；
- 超出容量时，要先拿到 `dummy.prv`，再从哈希表和链表中删除它；
- 插入头部时，更新指针的顺序不能写错；
- 哈希表中存的是 `key -> node`，不是 `key -> value`，否则无法在 `O(1)` 时间内移动链表节点。
