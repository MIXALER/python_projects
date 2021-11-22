import sys
from typing import List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def genTree(arr):
    def gen(arr, i):
        if i < len(arr):
            tn = TreeNode(arr[i]) if arr[i] != -1 else None
            if tn is not None:
                tn.left = gen(arr, 2 * i + 1)
                tn.right = gen(arr, 2 * i + 2)
            return tn

    return gen(arr, 0)


def binaryTreePaths(root: TreeNode) -> List[str]:
    if not root:
        return []
    if not root.left and not root.right:
        return [str(root.val)]
    paths = []
    if root.left:
        for i in binaryTreePaths(root.left):
            paths.append(str(root.val) + '->' + i)
    if root.right:
        for i in binaryTreePaths(root.right):
            paths.append(str(root.val) + '->' + i)
    return paths


if __name__ == "__main__":
    arr = sys.stdin.readline().strip().split()
    arr = [int(i) for i in arr]
    gen_tree = genTree(arr)
    ret_str = binaryTreePaths(gen_tree)
    ret_all = []
    for i in ret_str:
        tmp = i.split("->")
        tmp = [int(i) for i in tmp]
        ret_all.append(tmp)
    ret = []
    min = 999999
    for i in ret_all:
        if i[-1] <= min:
            min = i[-1]
    for i in ret_all:
        if i[-1] == min:
            ret = i
    for i in range(len(ret)):
        if i == len(ret) - 1:
            print(ret[i], end='\n')
        else:
            print(ret[i], end=' ')
