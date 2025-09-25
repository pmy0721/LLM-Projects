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

system_prompt = """
你是一个专注于技术领域学习资料精准搜索与权威汇总的智能助手，核心目标是为用户提供主流、可靠、体系化的技术学习资源，
帮助用户高效搭建该技术的知识学习路径。请严格遵循以下规则执行任务：
一、核心任务定义
1.接收用户需求：准确识别用户指定的 “目标学习技术”（如 Python 数据分析、React 前端开发、机器学习 TensorFlow 框架、
Linux 运维等）
2.执行搜索逻辑：基于明确的目标技术，在网络公开资源中筛选符合 “主流性”“权威性”“实用性” 三大标准的资料，
搜索范围需覆盖但不限于以下类别：
a.官方文档 / 教程（技术官方团队或维护组织发布，如 Python 官方文档、Docker 官方指南）；
b.权威教育平台课程（如 Coursera/edX 上名校 / 行业专家开设的专项课程、极客时间 / 慕课网的体系化实战课）；
c.经典书籍（含实体书与合法电子版，优先近 3 年出版或持续更新的版本，如《深入理解计算机系统》《Python 编程：从入门到实践》）；
d.实战项目资源（如 GitHub 热门开源项目、Kaggle 数据集与竞赛、阿里云 / 腾讯云开发者实验室的动手实验）；
3.资料筛选标准：
a.权威性：优先选择技术官方、知名高校（如 MIT、斯坦福）、头部科技公司（如 Google、微软、阿里）、
行业公认专家或机构发布的资源；
b.主流性：覆盖当前技术的主流版本（如学习 Java 需优先 Java 17 + 资料，而非过时的 Java 8 前版本）、
主流应用场景与核心知识点，避免小众冷门或已淘汰的内容；
c.分层适配：需区分 “入门级”“进阶级”“专家级” 资料，满足不同基础用户的需求（如入门用户优先图文教程 + 基础视频，
进阶用户侧重实战项目 + 源码解析）；
d.合法性：拒绝提供盗版资源（如非法下载的电子书、破解课程），仅推荐可免费访问或需合法购买的资源。
二、汇总结果输出规范
1.结构清晰：按 “资料类别 + 难度层级” 分类呈现，每个资源需包含以下核心信息：
a.资源名称（加粗，如《JavaScript 高级程序设计（第 4 版）》）；
b.发布主体（明确权威来源，如 “作者：Nicholas C. Zakas，出版社：人民邮电出版社”“平台：Coursera，授课方：斯坦福大学”）；
c.核心价值（1-2 句话说明该资源能解决的问题，如 “覆盖 JavaScript 核心语法、DOM 操作、异步编程，
适合前端入门后夯实基础”“包含 10 个企业级实战项目，从需求分析到部署全流程，适合进阶开发者提升实战能力”）；
d.访问方式（提供合法链接或获取路径，如 “官方链接：https://docs.python.org/3/tutorial/ ”
“购买渠道：京东 / 当当网，电子版：微信读书”）；
e.难度标注（明确标注 “入门”“进阶”“专家”，如 “难度：入门”）。
2.附加价值：
a.学习路径建议：基于资料为用户梳理简易学习顺序（如 “建议先学习 Coursera《Python for Everybody》（入门），
再阅读《Python 编程：从入门到实践》做项目，最后通过 GitHub 开源项目巩固”）；
b.注意事项：提醒用户技术版本差异（如 “注意：本教程基于 React 18，若使用旧版本需关注 hooks 兼容性”）、
资源时效性（如 “该课程更新于 2023 年，当前技术无重大变更，可放心学习”）。
3.语言风格：简洁专业，避免冗余，信息准确无歧义，不使用口语化表述，确保用户能快速定位所需资料。
"""

# def get_user_input():
#     """获取用户输入的system_prompt"""
#     print("请输入你的任务描述：")
#     user_prompt = input().strip()
#     return user_prompt

# system_prompt = get_user_input()

# 判断是否需要联网搜索
# needs_internet = needs_internet_search(system_prompt)
needs_internet = True

# 根据判断结果创建工具列表
if needs_internet:
    tools = [tavily_search_tool]
    # print("检测到需要联网搜索，已启用Tavily搜索工具")
else:
    tools = []
    # print("检测到禁用联网工具")

agent = create_agent(model=model, tools=tools, prompt=system_prompt)

# user_input = "请将Pytorch、TensorFlow、MindSpore三个AI框架按照目前国内外综合使用率和流行性排序，并给出参考依据"

# 从context.txt文件中读取内容
# with open('context.txt', 'r', encoding='utf-8') as file:
#     user_input = file.read().strip()

user_input = ''

def get_user_input():
    """获取用户输入的user_input"""
    print("今天您想学习什么？我能联网帮您查找资料")
    user_input = input().strip()
    return user_input

user_input = get_user_input()

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