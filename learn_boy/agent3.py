import asyncio
from typing import Literal, TypedDict, Annotated, List, Dict, Any
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
# 修正 ToolNode 的导入路径
from langchain.agents.tool_node import ToolNode
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI # 从 langchain_openai 导入
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.utils.function_calling import convert_to_openai_tool
from dotenv import load_dotenv
import json
import os

# 加载环境变量
load_dotenv()

# --- 1. 定义状态结构 ---
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] # 用于存储对话历史
    user_profile: Dict[str, str] # 存储用户背景信息 {"level": ..., "career": ...}
    current_topic: str # 当前学习的知识点
    knowledge_links: List[str] # 知识连接信息
    # 可以根据需要添加更多状态字段

# --- 2. 定义工具 ---

@tool
def get_user_background_tool(query: str) -> Dict[str, str]:
    """
    从用户输入中分析并提取其专业水平和职业背景。
    这个工具会分析消息历史和当前输入，更新状态中的 user_profile。
    """
    # 从环境变量获取 DeepSeek API 密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 环境变量未设置")
    llm = ChatOpenAI(model="deepseek-chat", temperature=0.0, base_url="https://api.deepseek.com", api_key=api_key)
    prompt = f"""
    请从以下用户输入中分析并提取出其专业水平和职业背景。
    专业水平选项: ['0基础', '有基础', '进阶', '专家']
    职业选项: ['学生', '程序员', '商业人士', '设计师', '教师', '其他']
    如果信息不明确，请返回 {{"level": "unknown", "career": "unknown"}}。
    用户输入: {query}
    请以 JSON 格式返回结果。
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        result = json.loads(response.content.strip())
        return result
    except json.JSONDecodeError:
        return {"level": "unknown", "career": "unknown"}

@tool
def identify_topic_tool(query: str) -> str:
    """
    从用户查询中识别出核心知识点。
    """
    # 从环境变量获取 DeepSeek API 密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 环境变量未设置")
    llm = ChatOpenAI(model="deepseek-chat", temperature=0.0, base_url="https://api.deepseek.com", api_key=api_key)
    prompt = f"请从以下用户问题中提取出核心学习知识点，只返回知识点名称，不要其他内容。用户问题: {query}"
    response = llm.invoke([HumanMessage(content=prompt)])
    topic = response.content.strip()
    return topic

@tool
def adjust_explanation_depth_tool(topic: str, level: str) -> str:
    """
    根据用户水平生成不同深度的解释。
    """
    # 从环境变量获取 DeepSeek API 密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 环境变量未设置")
    llm = ChatOpenAI(model="deepseek-chat", temperature=0.3, base_url="https://api.deepseek.com", api_key=api_key)
    prompt = f"""
    请为 '{topic}' 提供解释，要求如下：
    - 用户水平: {level}
    - 如果水平是 '0基础'，请使用比喻和故事化表达。
    - 如果水平是 '有基础'，请结合生活化例子和基本原理。
    - 如果是 '进阶' 或 '专家'，请提供更深入的技术细节。
    - 如果水平是 'unknown'，请提供中等深度的解释。
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

@tool
def provide_contextual_examples_tool(topic: str, career: str) -> str:
    """
    根据用户职业提供相关例子。
    """
    # 从环境变量获取 DeepSeek API 密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 环境变量未设置")
    llm = ChatOpenAI(model="deepseek-chat", temperature=0.3, base_url="https://api.deepseek.com", api_key=api_key)
    prompt = f"""
    请为 '{topic}' 提供例子，要求如下：
    - 用户职业: {career}
    - 如果职业是 '程序员'，请提供代码实例或系统架构类比。
    - 如果职业是 '商业人士'，请提供商业模式或市场竞争中的例子。
    - 如果职业是 '学生'，请提供校园生活或学习场景应用。
    - 如果职业是 '其他' 或 'unknown'，请提供通用例子。
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

@tool
def make_knowledge_connections_tool(topic: str, previous_topics: List[str]) -> List[str]:
    """
    生成知识连接信息。这里使用 LLM 来模拟知识连接的生成。
    """
    # 从环境变量获取 DeepSeek API 密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 环境变量未设置")
    llm = ChatOpenAI(model="deepseek-chat", temperature=0.1, base_url="https://api.deepseek.com", api_key=api_key)
    previous_topics_str = ", ".join(previous_topics)
    prompt = f"""
    请为知识点 '{topic}' 提供知识连接分析，基于以下信息：
    - 用户之前问过的知识点: {previous_topics_str}

    分析内容应包括：
    1. 与之前知识点的联系 (如果有的话)
    2. 学习此知识点前建议了解的前置知识
    3. 该概念的历史或最新研究方向 (如果适用)

    请返回一个包含连接信息的列表，例如 ["联系1", "建议1", "历史/前沿1"]。
    如果无法找到联系或建议，请返回一个包含通用信息的列表。
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        connections = json.loads(response.content.strip())
        return connections
    except json.JSONDecodeError:
        # 如果 LLM 没有返回 JSON，将其内容作为单个列表项返回
        return [response.content]

@tool
def update_user_profile_tool(new_info: Dict[str, Any]) -> str:
    """
    模拟更新用户画像的过程。实际应用中，这里会写入数据库。
    """
    print(f"模拟更新用户画像到数据库: {new_info}")
    return f"用户画像更新完成: {new_info}"

# --- 3. 定义 Agent 逻辑节点 ---
async def agent_node(state: State, config: RunnableConfig):
    """
    Agent 的核心逻辑节点。
    它根据当前状态和消息，决定下一步是生成 AI 回复还是调用工具。
    """
    messages = state["messages"]
    user_profile = state.get("user_profile", {})
    current_topic = state.get("current_topic", "")

    # 从环境变量获取 DeepSeek API 密钥
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 环境变量未设置")
    # 初始化 LLM - 使用 DeepSeek
    llm = ChatOpenAI(model="deepseek-chat", temperature=0.3, base_url="https://api.deepseek.com", api_key=api_key)

    # 定义工具
    tools = [
        get_user_background_tool,
        identify_topic_tool,
        adjust_explanation_depth_tool,
        provide_contextual_examples_tool,
        make_knowledge_connections_tool,
        update_user_profile_tool,
    ]

    # 定义提示模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        你是一个智能学习助手。你的任务是帮助用户学习。请遵循以下步骤：
        1.  如果用户尚未提供其专业水平和职业背景，请先询问并获取这些信息。
        2.  识别用户想了解的知识点。
        3.  根据用户的水平调整解释深度。
        4.  结合用户的背景提供适配的例子。
        5.  进行知识连接，提及与之前知识的联系、前置知识或前沿动态。

        请使用提供的工具来完成这些任务。
        """),
        MessagesPlaceholder(variable_name="messages"),
    ])

    # 绑定工具到 LLM
    llm_with_tools = llm.bind_tools([convert_to_openai_tool(t) for t in tools])

    # 构建链
    chain = prompt | llm_with_tools

    # 调用链
    response = await chain.ainvoke(state, config)

    return {"messages": [response]}


# --- 4. 构建图 ---
def build_graph():
    # 工具列表
    tools = [
        get_user_background_tool,
        identify_topic_tool,
        adjust_explanation_depth_tool,
        provide_contextual_examples_tool,
        make_knowledge_connections_tool,
        update_user_profile_tool,
    ]

    # 创建 ToolNode - 使用修正后的导入路径
    tool_node = ToolNode(tools, handle_tool_errors=True)

    # 创建状态图
    workflow = StateGraph(State)

    # 添加节点
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # 定义条件边：决定下一步是调用工具还是结束
    def should_continue(state: State) -> Literal["tools", "__end__"]:
        messages = state["messages"]
        last_message = messages[-1]
        # 如果最后一条消息是 AIMessage 并且包含 tool_calls，则需要调用工具
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        # 否则，结束循环
        return "__end__"

    # 添加边
    workflow.add_conditional_edges(
        "agent",
        should_continue,
    )
    workflow.add_edge("tools", "agent") # 工具执行后回到 agent

    # 设置入口点
    workflow.set_entry_point("agent")

    # 编译图
    return workflow.compile()

# --- 5. 主程序 ---
async def main():
    # 检查环境变量 - 检查 DeepSeek 的密钥
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误: 未找到 DEEPSEEK_API_KEY 环境变量。请设置后再运行。")
        return

    app = build_graph()

    # 初始状态
    initial_state = State(
        messages=[SystemMessage(content="你好！我是你的智能学习助手。请告诉我你想了解什么知识，以及你的专业背景（如：我是程序员，我有基础等）。")],
        user_profile={},
        current_topic="",
        knowledge_links=[]
    )

    print("智能学习助手已启动！请输入您想了解的知识。")
    print("(例如: '我想学习神经网络', '我是一个初学者', '我是个程序员')")

    # 初始化对话
    try:
        async for event in app.astream(initial_state):
            if "agent" in event:
                ai_message = event["agent"]["messages"][-1]
                if isinstance(ai_message, AIMessage) and not ai_message.tool_calls:
                    print(f"\n助手: {ai_message.content}")
    except Exception as e:
        print(f"初始化对话时出错: {e}")
        import traceback
        traceback.print_exc()
        return

    while True:
        try:
            user_input = input("\n您: ")
            if user_input.lower() in ["退出", "exit", "quit"]:
                print("感谢使用智能学习助手！再见！")
                break

            if not user_input.strip():
                print("请输入有效的问题...")
                continue

            # 向图发送用户输入和当前状态
            # astream 会持续输出直到图结束
            final_state = None
            async for event in app.astream({"messages": [HumanMessage(content=user_input)]}, stream_mode="values"):
                 final_state = event # 获取最终状态

            # 从最终状态中获取并打印助手的最终回复
            if final_state and final_state["messages"]:
                last_message = final_state["messages"][-1]
                if isinstance(last_message, AIMessage) and not last_message.tool_calls: # 确保是最终回复，而非工具调用
                    print(f"\n助手: {last_message.content}")
                # 如果最后是工具调用，说明 agent 逻辑可能有问题，或者工具执行失败

        except KeyboardInterrupt:
            print("\n\n程序被用户中断。")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")
            import traceback
            traceback.print_exc()
            break

if __name__ == "__main__":
    asyncio.run(main())