import sys

if __name__ == "__main__":
    n = input()
    n = int(n)

    arr = sys.stdin.readline().strip().split()
    arr = [int(i) for i in arr]

    scope = []

    for i in range(len(arr)):
        if i == 0:
            scope.append([0, arr[i]])
        elif i == (len(arr) - 1):
            scope.append([100 * i - arr[i], 100 * i])
        else:
            scope.append([100 * i - arr[i], i * 100])
            scope.append([i * 100, 100 * i + arr[i]])

    sum = 0

    j = 0
    while j <= len(scope) - 2:

        first = scope[j][1]
        k = j + 1
        second_min = scope[k][0]
        while k <= len(scope) - 1:
            if scope[k][0] < second_min:
                second_min = scope[k][0]
            k = k + 1
        if first >= second_min:
            j = j + 2
        else:
            j = j + 2
            sum += second_min - first
    print(sum)
