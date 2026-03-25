def bijiao(s1: str, s2: str) -> bool:
    "比较两个版本号，若s1>s2,则返回True"
    a1, a2 = list(map(int, s1.split('.'))), list(map(int, s2.split('.')))
    # 补全3位版本数
    while len(a1) < 3:
        a1.append(0)
    while len(a2) < 3:
        a2.append(0)
    # print(a1)
    # print(a2)

    # 比较
    for i in range(3):
        if a1[i] > a2[i]:
            return True
        elif a1[i] < a2[i]:
            return False
        else:
            continue
    return False


def sort(arr):
    "冒泡排序(优化)"
    for i in range(len(arr) - 1):
        # 标志位
        flag = False
        for j in range(len(arr) - i - 1):
            if bijiao(arr[j], arr[j + 1]):
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                flag = True
        if not flag:
            break
    print(arr)


def main():
    arr1 = ["2.1.3", "1.10.2", "1.9.1", "3.0.0", "1.10.1"]
    arr2 = ["9.1", "8.10.5", "8.10", "6.0.1", "5.10"]
    arr3 = ["1.1.3", "2.10.2", "3.9.1", "4.0.0", "5.10.1"]
    sort(arr2)


if __name__ == "__main__":
    main()
    # print(bijiao('2.1.3', '1.10.1'))
