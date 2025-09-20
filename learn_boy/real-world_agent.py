import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass


# 加载.env文件中的环境变量
load_dotenv()          # os.environ["DEEPSEEK_API_KEY"]

def get_weather_for_location(city: str) -> str:  # (1)!
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

USER_LOCATION = {
    "1": "Florida",
    "2": "SF"
}

@tool
def get_user_location(config: RunnableConfig) -> str:
    """Retrieve user information based on user ID."""
    # 从配置中提取user_id
    runtime = config["configurable"].get("__pregel_runtime")
    if runtime and hasattr(runtime, "context"):
        user_id = runtime.context.get("user_id")
    else:
        print("未找到user_id")
    return USER_LOCATION[user_id]

system_prompt = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. 

If you can tell from the question that they mean whereever they are, 

use the get_user_location tool to find their location."""

model = init_chat_model(
    "deepseek:deepseek-chat",
    temperature=0
)

@dataclass
class WeatherResponse:
    conditions: str
    punny_response: str

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    prompt=system_prompt,
    tools=[get_user_location, get_weather_for_location],
    response_format=WeatherResponse,
    checkpointer=checkpointer
)

config = {"configurable": {"thread_id": "1"}}
context = {"user_id": "2"}
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=context
)

print(response['structured_response'])

response = agent.invoke(
    {"messages": [{"role": "user", "content": "thank you!"}]},
    config=config,
    context=context
)

print(response['structured_response'])


# final_response = next(
#     msg.content for msg in reversed(ai_msg["messages"])
#     if msg.__class__.__name__ == "AIMessage"
# )
# print(final_response)