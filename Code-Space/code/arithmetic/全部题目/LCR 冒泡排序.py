def mao_pao(arr):
    ...
    for i in range(len(arr)):
        flag=False # 标志位初始化为false，若没有排序则说明当前已有序
        
        # 不断缩小冒泡范围
        for j in range(0,len(arr)-i-1):
            if arr[j]>arr[j+1]:
                arr[j],arr[j+1]=arr[j+1],arr[j]
                flag=True
            
            # 当前已有序
            if not flag:
                break
        
        return arr

def bubble_sort_optimized(arr):
    n = len(arr)
    for i in range(n):
        swapped = False 
        # n-i-1 是因为每轮后面 i 个元素已经就位
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True # 发生了交换
        # 如果一整轮都没有发生交换，说明数组已排序，提前退出
        if not swapped:
            break
    return arr

# 示例
# print(bubble_sort_optimized([11, 12, 22, 25, 34, 64, 90])) # 已排序序列
# print(bubble_sort_optimized([64, 34, 25, 12, 22, 11, 90])) # 未排序序列
