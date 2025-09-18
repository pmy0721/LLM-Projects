import os
from langchain.agents import create_agent

if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "***"

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="deepseek:deepseek-chat",
    tools=[get_weather],
    prompt="You are a helpful assistant",
)

# Run the agent
ai_msg = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

final_response = next(
    msg.content for msg in reversed(ai_msg["messages"])
    if msg.__class__.__name__ == "AIMessage"
)
print(final_response)

