#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain 0.x 与 1.x 语法对比示例
作者：资深全栈架构师
日期：2025-12-22
"""

# ============================================================================
# 1. 导入路径对比
# ============================================================================

print("=== LangChain 导入路径对比 ===\n")

# 0.x 版本导入方式（你现在使用的）
print("🔹 0.x 版本导入方式：")
print("""
# 链式操作
from langchain.chains import ConversationChain, LLMChain
from langchain.chains.conversation.memory import ConversationBufferMemory

# 语言模型
from langchain_community.llms import Tongyi
from langchain_community.chat_models import ChatOpenAI

# 提示模板
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
""")

# 1.x 版本导入方式
print("\n🔹 1.x 版本导入方式：")
print("""
# 链式操作（新方式）
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 语言模型（新路径）
from langchain_community.chat_models import ChatTongyi
from langchain_openai import ChatOpenAI

# 提示模板（新方式）
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# 记忆管理（新方式）
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
""")

# ============================================================================
# 2. 链式操作对比
# ============================================================================

print("\n=== 链式操作语法对比 ===\n")

# 0.x 版本的链式操作
print("🔹 0.x 版本链式操作：")
print("""
# 传统链式操作
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
chain = ConversationChain(llm=llm, memory=memory)
response = chain.run("你好")
""")

# 1.x 版本的链式操作
print("\n🔹 1.x 版本链式操作：")
print("""
# 新的 Runnable 接口
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# 使用 Runnable 组合
prompt = ChatPromptTemplate.from_template("回答：{question}")
chain = prompt | llm
response = chain.invoke({"question": "你好"})
""")

# ============================================================================
# 3. 记忆管理对比
# ============================================================================

print("\n=== 记忆管理语法对比 ===\n")

# 0.x 版本记忆管理
print("🔹 0.x 版本记忆管理：")
print("""
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

# 在链中使用
chain = ConversationChain(llm=llm, memory=memory)
""")

# 1.x 版本记忆管理
print("\n🔹 1.x 版本记忆管理：")
print("""
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 创建消息历史
chat_history = ChatMessageHistory()

# 使用 RunnableWithMessageHistory
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)
""")

# ============================================================================
# 4. 工具集成对比
# ============================================================================

print("\n=== 工具集成语法对比 ===\n")

# 0.x 版本工具集成
print("🔹 0.x 版本工具集成：")
print("""
from langchain.agents import initialize_agent
from langchain.tools import Tool

# 定义工具
tools = [
    Tool(
        name="Calculator",
        func=lambda x: eval(x),
        description="用于数学计算"
    )
]

# 初始化代理
agent = initialize_agent(tools, llm, agent="zero-shot-react-description")
""")

# 1.x 版本工具集成
print("\n🔹 1.x 版本工具集成：")
print("""
from langchain.agents import create_tool_calling_agent
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser

# 使用新的工具调用代理
agent = create_tool_calling_agent(llm, tools, prompt)
""")

print("\n" + "="*60)
print("✅ 语法对比完成！详细说明见下文分析")
print("="*60)