# 可以引⼊的库和版本相关请参考 “环境说明”

from typing import Dict


def func(s: str, dic):
    for i in range(len(s)):
        if s[i] in dic:
            dic[s[i]] = dic[s[i]] + 1
        else:
            dic[s[i]] = 1
        # print(dic)


# 本题面试官已设置测试用例
def isMerge(s: str, part1: str, part2: str) -> bool:
    dic1 = {}
    func(s, dic1)
    dic2 = {}
    func(part1, dic2)
    func(part2, dic2)
    print(dic1)
    print(dic2)
    for key in dic1.keys():
        if key not in dic2.keys():
            # print(key)
            return False
        else:
            if dic1[key] != dic2[key]:
                # print(key)
                return False
    return True


def main():

    s = "showmebug"
    part1 = "somb"
    part2 = "hweug"
    ans = isMerge(s, part1, part2)
    print(ans)


main()
