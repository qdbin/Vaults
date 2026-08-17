def main(s: str) -> str:
    stack = []
    cur_str = ""
    cur_num = 0
    for c in s:
        if c.isdigit():
            cur_num = cur_num * 10 + int(c)  #!拼接多位数
        elif ch == '[':
            stack.append((cur_str, cur_num))
            cur_num = 0
        elif ch == ']':
            pre_str, num = stack.pop()
            cur_str = pre_str + cur_str * num
        else:
            cur_str += chr
    return cur_str
