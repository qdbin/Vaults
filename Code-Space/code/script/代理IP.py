"""
    介绍：获取批量有效的代理ip
    视频链接：https://v.douyin.com/UgeFqahEUeY/
    当前进度：抓取对应页面的所有ip，但没有过滤是否有效
"""
import json
import requests
import re
head={'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36'}
base_url="https://www.kuaidaili.cn/free/dps/2"
try:
    res=requests.get(url=base_url,headers=head)
except Exception as e:
    print(f'发送了错误====>{e}')
# print(res)
# print(res.text)

re_res=re.findall('const\s*fpsList\s*=\s*(\[.*?\])',res.text)
re_list=json.loads(re_res[0])
# print(re_list)
for item in re_list:
    ip=item['ip']+':'+item['port']
    print(ip)

'http:httpbin.org/ip'
# print(re_res)