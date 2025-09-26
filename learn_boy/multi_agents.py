import os
import sys
import time
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

learn_plan_agent_system_prompt = """
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
请注意，在回答的开头需声明本学习计划由学习计划生成智能体生成。
"""

learn_data_agent_system_prompt = """
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
请注意，在回答的开头需声明本学习资料汇总由学习资料查找智能体生成。
"""

learn_plan_agent = create_agent(model=model, tools=[tavily_search_tool], prompt=learn_plan_agent_system_prompt)

learn_data_agent = create_agent(model=model, tools=[tavily_search_tool], prompt=learn_data_agent_system_prompt)

@tool
def learn_plan_agent_tool(query: str):
    """根据用户输入的学习目标，生成学习计划"""
    print('正在调用「计划生成智能体」生成结果...')
    result = learn_plan_agent.invoke({
        "messages":[{"role": "user", "content": query}]
    })
    return result["messages"][-1].content

@tool
def learn_data_agent_tool(query: str):
    """根据用户输入的学习目标，搜索相关的学习资料"""
    print('正在调用「资料搜索智能体」生成结果...')
    result = learn_data_agent.invoke({
        "messages":[{"role": "user", "content": query}]
    })
    return result["messages"][-1].content

# 主控智能体系统提示词
master_agent_system_prompt = """
你是一个智能学习助手主控系统，负责协调和编排多个专业子智能体来为用户提供全面的学习支持。
你拥有以下两个专业子智能体工具：

1. **学习计划生成智能体** - 专门负责根据用户的学习目标生成详细、可执行的学习计划
2. **学习资料搜索智能体** - 专门负责搜索和推荐权威、优质的学习资源

## 核心职责与工作流程

### 任务分析与智能编排
当用户提出学习相关需求时，你需要：

1. **需求理解**：准确理解用户的学习目标、当前水平、时间安排等关键信息
2. **任务分解**：明确用户需求，仅调用必需的子智能体工具（如学习计划生成智能体、学习资料搜索智能体）
3. **智能编排**：根据需求复杂度和用户意图，决定子智能体的调用顺序和方式

### 结果整合

**场景1：仅需学习计划**
- 用户明确表示只需要学习计划或学习路线
- 直接调用学习计划生成智能体
- 直接输出学习计划生成智能体生成的结果

**场景2：仅需学习资料**
- 用户明确表示只需要学习资源推荐
- 直接调用学习资料搜索智能体
- 直接输出学习资料搜索智能体生成的结果

**场景3：需要完整学习方案（推荐）**
- 用户提出学习某项技术的综合需求
- 先调用学习计划生成智能体制定计划
- 再调用学习资料搜索智能体寻找配套资源
- 直接将两者结果拼接后输出，形成"学习计划+配套资源"的完整方案

**场景4：资料优先的学习方案**
- 用户更关注学习资源的质量和权威性
- 先调用学习资料搜索智能体获取优质资源
- 直接将两者结果拼接后输出，形成"学习计划+配套资源"的完整方案

"""

# 创建主控智能体
master_agent = create_agent(
    model=model, 
    tools=[learn_plan_agent_tool, learn_data_agent_tool], 
    prompt=master_agent_system_prompt
)

# 主控智能体调用函数（流式输出版本）
def run_master_agent_stream(user_query: str):
    """
    主控智能体入口函数（流式输出版本）
    
    Args:
        user_query (str): 用户的学习需求查询
        
    Returns:
        None: 直接流式输出到控制台
    """
    try:
        # 使用stream方法进行流式调用
        stream = master_agent.stream({
            "messages": [{"role": "user", "content": user_query}]
        })
        
        # 流式输出每个chunk
        for chunk in stream:
            # 检查chunk中是否包含agent的输出
            if "agent" in chunk:
                agent_output = chunk["agent"]
                if "messages" in agent_output and agent_output["messages"]:
                    latest_message = agent_output["messages"][-1]
                    if hasattr(latest_message, 'content') and latest_message.content:
                        # 逐字符输出，模拟打字机效果
                        for char in latest_message.content:
                            print(char, end='', flush=True)
                            time.sleep(0.02)  # 控制打字速度，可调整
                        print()  # 输出完成后换行
            # 检查是否有工具调用的输出
            elif "tools" in chunk:
                tools_output = chunk["tools"]
                if "messages" in tools_output and tools_output["messages"]:
                    for message in tools_output["messages"]:
                        if hasattr(message, 'content') and message.content:
                            # 逐字符输出，模拟打字机效果
                            for char in message.content:
                                print(char, end='', flush=True)
                                time.sleep(0.02)  # 控制打字速度，可调整
                            print()  # 输出完成后换行
                            
    except Exception as e:
        print(f"处理请求时出现错误: {str(e)}")

# 原有的非流式版本保留
def run_master_agent(user_query: str):
    """
    主控智能体入口函数（非流式版本）
    
    Args:
        user_query (str): 用户的学习需求查询
        
    Returns:
        str: 整合后的完整回答
    """
    try:
        result = master_agent.invoke({
            "messages": [{"role": "user", "content": user_query}]
        })
        return result["messages"][-1].content
    except Exception as e:
        return f"处理请求时出现错误: {str(e)}"

# 为了向后兼容，保留原有的agent变量
agent = master_agent

def get_user_input():
    """获取用户输入的user_input"""
    print("今天您想学习什么？")
    user_input = input().strip()
    return user_input

# 使用流式输出版本
user_input = get_user_input()
run_master_agent_stream(user_input)

# 使用原始输出
# response = run_master_agent(user_input)
# print(response)
