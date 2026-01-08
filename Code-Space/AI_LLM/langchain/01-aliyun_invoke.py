from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Tongyi

# 创建一个内存记忆对象
memory= ConversationBufferMemory(return_message=True)

def get_response(prompt,api_key):
    model = Tongyi(model="qwen-max",api_key=api_key)
    chain= ConversationChain(llm=model,memory=memory)
    
    #发送用户的请求
    response = chain.invoke({"input":prompt})
    return response["response"]

if __name__=='__main__':
    api_key="sk-6cc232b651b54347b2f1938a76e0898c"
    print(get_response("请用python输出15-30",api_key=api_key))