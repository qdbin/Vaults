"""
    deepseek生成的脚本，可以根据出发地和目的地查询出对应的车次和余票
"""
import requests
import json
from datetime import datetime

class TrainTicketQuery:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36',
            'Referer': 'https://kyfw.12306.cn/otn/leftTicket/init',
            'X-Requested-With': 'XMLHttpRequest',
            'Sec-Ch-Ua': '"Google Chrome";v="137", "Chromium";v="137", "Not=A?Brand";v="24"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Cache-Control': 'no-cache',
        }
        # 初始化必要的cookies和动态参数
        self._init_session()

    def _init_session(self):
        """初始化会话，获取必要的cookies"""
        init_url = "https://kyfw.12306.cn/otn/leftTicket/init"
        self.session.get(init_url, headers=self.headers)
        
        # 获取动态JS (实际使用时可能需要处理)
        dynamic_js_url = "https://kyfw.12306.cn/otn/dynamicJs/qgrqrrp"
        self.session.get(dynamic_js_url, headers=self.headers)

    def query_tickets(self, from_station, to_station, date):
        """查询车票信息"""
        url = "https://kyfw.12306.cn/otn/leftTicket/queryU"
        
        params = {
            'leftTicketDTO.train_date': date,
            'leftTicketDTO.from_station': from_station,
            'leftTicketDTO.to_station': to_station,
            'purpose_codes': 'ADULT'
        }

        try:
            response = self.session.get(
                url,
                params=params,
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') and data.get('httpstatus') == 200:
                return self._parse_ticket_data(data['data']['result'])
            else:
                print("查询失败:", data.get('messages', '未知错误'))
                return []
                
        except requests.exceptions.RequestException as e:
            print("请求出错:", str(e))
            return []
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print("解析响应数据出错:", str(e))
            return []

    def _parse_ticket_data(self, train_list):
        """解析车次数据"""
        tickets = []
        
        for train in train_list:
            info = train.split('|')
            
            # 解析车次信息 (索引可能需要根据实际情况调整)
            ticket = {
                'train_no': info[3],          # 车次
                'start_station': info[6],     # 始发站
                'end_station': info[7],       # 终点站
                'from_station': info[4],      # 出发站
                'to_station': info[5],        # 到达站
                'start_time': info[8],        # 出发时间
                'arrive_time': info[9],       # 到达时间
                'duration': info[10],          # 历时
                'can_buy': info[11] == 'Y',   # 能否购买
                'train_class': info[21],      # 车次类型
                'seats': {
                    'business': info[32] or '--',  # 商务座/特等座
                    'first_class': info[31] or '--',  # 一等座
                    'second_class': info[30] or '--',  # 二等座
                    'soft_sleeper': info[23] or '--',  # 软卧
                    'hard_sleeper': info[28] or '--',  # 硬卧
                    'soft_seat': info[24] or '--',  # 软座
                    'hard_seat': info[29] or '--',  # 硬座
                    'no_seat': info[26] or '--',  # 无座
                }
            }
            tickets.append(ticket)
            
        return tickets

    def print_tickets(self, tickets):
        """打印车票信息"""
        for ticket in tickets:
            print(f"\n车次: {ticket['train_no']} ({ticket['train_class']})")
            print(f"时间: {ticket['start_time']} → {ticket['arrive_time']} (历时: {ticket['duration']})")
            print(f"区间: {ticket['from_station']} → {ticket['to_station']}")
            print("座位情况:")
            print(f"  商务/特等: {ticket['seats']['business']} 一等: {ticket['seats']['first_class']} 二等: {ticket['seats']['second_class']}")
            print(f"  软卧: {ticket['seats']['soft_sleeper']} 硬卧: {ticket['seats']['hard_sleeper']}")
            print(f"  硬座: {ticket['seats']['hard_seat']} 无座: {ticket['seats']['no_seat']}")
            print("-" * 60)

if __name__ == '__main__':
    # 示例查询: 北京东(BOP)到天津(TJP)，2025-07-19的车票
    query = TrainTicketQuery()
    
    from_station = 'BOP'  # 北京东站代码
    to_station = 'TJP'    # 天津站代码
    query_date = '2025-08-15'  # 查询日期
    
    tickets = query.query_tickets(from_station, to_station, query_date)
    query.print_tickets(tickets)