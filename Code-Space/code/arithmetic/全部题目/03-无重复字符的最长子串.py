"""
    链接：https://leetcode.cn/problems/longest-substring-without-repeating-characters/description/
    
    思想：滑动窗口
        - 使用滑动窗口，当重复字符存在，使用max()确保左窗口滑动前进，同时也要确保dict左边界重新赋值
        - 无重复长度使用cur_len=right-left+1,同时
        - for循环的循环逻辑不能被改变（改变i没有用）
    
    实现：
        ！！！每次遍历字符，若出现char in dic中，都要使用max（）选择是否要更新left左窗口，然后务必更新字典的字符下标
    注意：
        - right右窗口自动遍历
        - 若遍历的当前字符存在字典中，务必更新最新字符的下标！！！，更新left下标根据max（）判断是否更新
        - 字典只增不少，每次出现重复出现字符都需要更新下标
        - 少用:=（每次都会重新赋值）
"""



class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        # Init 左窗口下标、Init '字符':'下标'字典、Init 最大子串长度
        left,dic,max_len=0,{},0

        # 遍历更新右窗口下标
        for right in range(len(s)):
            # 如果当前字符存在字典中，先更新left左窗口，再字典右窗口的下标
            if s[right] in dic:
                # 更新左窗口下标（max确保左窗口只进不退）
                left=max(left,dic[s[right]]+1)      # 如当前left为10，但字典中的下标为2（则说明这个字符已不在当前的滑动窗口中了）
                dic[s[right]]=right     #!不要忘记更新当前'字符'的字典'下标'

            else:
                dic[s[right]]=right
            
            # 更新最大子串长度
            if (right-left+1>max_len):
                max_len=right-left+1

        return max_len

if __name__=='__main__':
    s=str("abba")
    res=Solution().lengthOfLongestSubstring(s)
    print(res)