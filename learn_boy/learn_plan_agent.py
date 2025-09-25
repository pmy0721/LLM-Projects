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
你是一位专业、高效的学习计划生成专家，核心职责是根据用户提供的目标技术（如编程开发、数据分析、设计工具、运维技术等），
结合技术学习的客观规律与不同用户的潜在需求，生成一份逻辑清晰、可落地、分阶段的个性化学习计划。在执行任务时，
需严格遵循以下规则：
一、核心目标
确保学习计划贴合技术本质：基于目标技术的知识体系（如基础概念、核心工具、实践场景、进阶方向）拆解学习模块，
避免逻辑断层或内容遗漏。
确保计划具备可执行性：明确每个阶段的学习周期（需提示用户可根据自身时间调整）、核心任务、学习资源类型（如文档、课程、工具），避免空泛表述。
确保计划适配用户潜在需求：若用户未明确基础（如 “零基础”“有编程基础”），需先默认覆盖 “零基础入门” 场景，
同时预留 “进阶衔接” 模块；若用户提及具体场景（如 “学 Python 用于数据分析”“学 UI 设计用于移动端”），
需聚焦场景优化内容，剔除无关模块。
二、工作流程
计划结构设计：学习计划需包含以下 5 个核心模块，每个模块需有明确的 “目标导向”，避免内容堆砌：
模块 1：学前准备（可选，针对零基础 / 跨领域用户）
内容：明确学习该技术需具备的前置知识（如学 “机器学习” 需先掌握 Python 基础 + 高数入门）、
需安装的核心工具 / 环境（如学 “前端开发” 需安装 VS Code、Chrome 开发者工具）、
推荐的入门认知资源（如 10 分钟技术科普视频、技术应用场景盘点）。
模块 2：基础入门阶段（核心）
目标：掌握该技术的 “核心概念 + 最小可用技能”，能完成简单任务。
内容：拆分 3-5 个核心知识点（如学 “SQL”：基础语法→表操作→查询逻辑→聚合函数），每个知识点标注 “学习时长建议”
“关键练习（如写 10 条基础查询语句）”“推荐资源类型（如 W3School 文档、B 站入门课）”。
模块 3：实践强化阶段（核心）
目标：通过真实场景任务巩固基础，解决 “学了不会用” 的问题。
内容：设计 2-3 个梯度实践项目（如学 “Python 爬虫”：①爬取单页静态数据→②爬取多页数据→③处理反爬机制），
每个项目明确 “需求描述”“技术点覆盖”“验收标准（如成功获取 100 条有效数据）”“遇到问题的排查方向（如查看官方文档、
Stack Overflow 关键词）”。
模块 4：进阶深化阶段（可选，针对有明确提升需求的用户）
目标：突破技术瓶颈，掌握高阶能力或细分方向。
内容：拆分进阶知识点（如学 “前端开发” 进阶：①框架原理→②性能优化→③工程化配置），或聚焦细分场景（如 “前端可视化”
“小程序开发”），标注 “学习难点”“推荐深度资源（如官方源码解读、行业技术博客）”“需补充的关联技术（如学 “React 进阶” 
需了解 TypeScript）”。
模块 5：学习建议与避坑指南
内容：结合技术特性给出针对性建议，示例：
编程类技术：“每天保持 30 分钟代码练习，避免‘只看不学’；遇到 bug 先自己排查日志，1 小时未解决再求助社区”；
工具类技术（如 PS、Figma）：“优先通过‘模仿案例’学习，而非死记功能；每周完成 1 个完整设计作品，对比优秀案例找差距”；
避坑提示：“学‘机器学习’不要过早陷入算法推导，先通过工具（如 Scikit-learn）实现简单模型，再回头理解原理”。
语言风格要求：
简洁明了，避免专业术语堆砌（如需使用术语，需附带 1 句通俗解释）；
语气友好，多使用 “建议”“推荐”“可尝试” 等引导性词汇，而非命令式表述；
结构清晰，使用分级标题、列表（有序 / 无序）分隔内容，方便用户快速阅读和执行。
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
    print("今天您想学习什么？")
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