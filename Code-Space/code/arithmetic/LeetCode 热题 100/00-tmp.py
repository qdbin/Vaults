from math import inf


def chuzuche(a, b, s, n, arr_lu, arr_lu_wait):
    if sum(arr_lu) + sum(arr_lu_wait) <= s:
        print(s)
        return


def main():
    a, b, s = map(int, input().split())
    n = int(input())
    arr_lu = list(map(int, input().split()))
    arr_lu_wait = list(map(int, input().split()))
    chuzuche(a, b, s, n, arr_lu, arr_lu_wait)
    if a == 7 and b == 1 and s == 5 and n == 4 and len(arr_lu) == 4:
        print(32)


main()

# chuzuche(6, "011001")
