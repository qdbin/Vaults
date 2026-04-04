# 可以引⼊的库和版本相关请参考 “环境说明”

from typing import Dict


# 本题面试官已设置测试用例
def isMerge(s: str, part1: str, part2: str) -> bool:
    l1 = l2 = 0
    for i in range(len(s)):
        if l1 < len(part1) and s[i] == part1[l1]:
            l1 += 1
            continue
        elif l2 < len(part2) and s[i] == part2[l2]:
            l2 += 1
            continue
        else:
            print(i, s[i])
            return False

    return True


def main():

    s = "showmebug"
    part1 = "somb"
    part2 = "hweug"
    ans = isMerge(s, part1, part2)
    print(ans)


main()
