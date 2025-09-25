import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain_tavily import TavilySearch
from dataclasses import dataclass

# 加载.env文件中的环境变量
load_dotenv()          # os.environ["DEEPSEEK_API_KEY"]

model = init_chat_model(
    "deepseek:deepseek-chat",
    temperature=0
)

tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
    # include_answer=False,
    # include_raw_content=False,
    # include_images=False,
    # include_image_descriptions=False,
    # search_depth="basic",
    # time_range="day",
    # include_domains=None,
    # exclude_domains=None
)

# tool_msg = tavily_search_tool.invoke({"query": "什么是大模型？"})
# print(tool_msg)

def needs_internet_search(prompt):
    """
    判断给定的prompt是否需要联网搜索
    返回True表示需要联网，False表示不需要
    """
    # 需要联网搜索的关键词
    internet_keywords = [
        "最新", "当前", "目前", "现在", "今天", "实时", "最近",
        "流行度", "现状"
    ]
    
    # 不需要联网搜索的关键词（基于已有文本分析）
    local_keywords = [
        "上下文"
    ]
    
    prompt_lower = prompt.lower()
    
    # 如果包含明确的本地处理关键词，优先判断为不需要联网
    for keyword in local_keywords:
        if keyword in prompt_lower:
            return False
    
    # 检查是否包含需要联网的关键词
    for keyword in internet_keywords:
        if keyword in prompt_lower:
            return True
    
    # 默认情况下，返回False（禁用联网）
    return False

# system_prompt = """
# 你的任务是根据提供的上下文，提取出跟大模型有关的技术，并统计出现频率
# """

def get_user_input():
    """获取用户输入的system_prompt"""
    print("请输入你的任务描述：")
    user_prompt = input().strip()
    return user_prompt

system_prompt = get_user_input()

# 判断是否需要联网搜索
needs_internet = needs_internet_search(system_prompt)

# 根据判断结果创建工具列表
if needs_internet:
    tools = [tavily_search_tool]
    print("检测到需要联网搜索，已启用Tavily搜索工具")
else:
    tools = []
    print("检测到禁用联网工具")

agent = create_agent(model=model, tools=tools, prompt=system_prompt)

# user_input = "请将Pytorch、TensorFlow、MindSpore三个AI框架按照目前国内外综合使用率和流行性排序，并给出参考依据"

# 从context.txt文件中读取内容
# with open('context.txt', 'r', encoding='utf-8') as file:
#     user_input = file.read().strip()

user_input = ''

# 获取agent的完整响应
result = None
for step in agent.stream(
    {"messages": user_input},
    stream_mode="values",
):
    result = step

# 输出最终的AI消息
if result:
    result["messages"][-1].pretty_print()


