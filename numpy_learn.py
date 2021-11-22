import sys
from collections import defaultdict
from typing import List
from math import inf
import heapq


def countPaths(s: int, d: int, roads: List[List[int]]):
    # 使用临接矩阵表达图中节点与节点之间的距离
    graph = defaultdict(lambda: defaultdict(int))
    for node, node_next, cost in roads:
        graph[node - 1][node_next - 1] = cost
        graph[node_next - 1][node - 1] = cost

    # 使用数组表示到达各个节点的最短距离
    n = len(graph)
    dist = [inf for _ in range(n)]
    dist[s] = 0

    # 使用数组表示到达各个节点最短距离的路线数
    count = [0 for _ in range(n)]
    count[s] = 1

    # 使用最小堆维护到达各个节点的最短距离
    heap = []
    heapq.heappush(heap, (0, s))

    while heap:
        cost, node = heapq.heappop(heap)
        if cost > dist[node]:
            continue
        for node_next, cost in graph[node].items():
            if dist[node] + cost < dist[node_next]:
                dist[node_next] = dist[node] + cost
                count[node_next] = count[node]
                heapq.heappush(heap, (dist[node_next], node_next))
            elif dist[node] + cost == dist[node_next]:
                count[node_next] += count[node]
    return dist[d], count[d] % (10 ** 9 + 7)


if __name__ == "__main__":
    s, d = [int(k) for k in sys.stdin.readline().strip().split()]
    g = eval(sys.stdin.readline().strip())
    pair = countPaths(s - 1, d - 1, g)
    print(f'{pair[0]} {pair[1]}')
