"""
这个题目是病原体为1，正常为0，由0和1组成的字符串，每一个1每一秒都会向两边扩散，感染0，需要你返回最快多久能把所有的0都感染成1
"""


def main(n: int, s: str) -> int:
    s_one_cnt = s.count('1')
    if s_one_cnt == 0:
        return -1
    if s_one_cnt == n:
        return 0

    l, max_len, start_end = 0, 0, True
    for r in range(n):
        if s[r] == '1':
            max_len = max(max_len, r - l)
            l = r + 1
    time1 = max_len // 2 + 1 if max_len & 1 else max_len // 2
    first_one, last_one = s.index('1'), s[::-1].index('1')
    # print(f'max_len:{last_one}\n first_one:{first_one}\n last_one:{last_one}')
    return max(first_one, last_one, time1)


def start():
    T = int(input())
    for _ in range(T):
        n = int(input())
        s = input()
        ans = main(n, s)
        print(ans)


start()
# t=main(9,'010000101')
# print(t)
