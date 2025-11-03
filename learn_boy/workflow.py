import os
import getpass

from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=1.3,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # api_key="...",
    # other params...
)

# response = llm.invoke("晚饭吃了烧卖以后，胃有点不舒服，想知道是否有什么办法可以缓解不适")
# print(response.content)

# 结构化输出的模式
from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query that is optimized web search.")
    justification: str = Field(
        None, description="Why this query is relevant to the user's request."
    )


# Augment the LLM with schema for structured output
structured_llm = llm.with_structured_output(SearchQuery)

# Invoke the augmented LLM
output = structured_llm.invoke("我晚上吃完烧卖以后，现在有点烧心，我该如何缓解？")
print("结构化输出:", output)

# 定义一个工具
def multiply(a: int, b: int) -> int:
    return a * b

# Augment the LLM with tools
llm_with_tools = llm.bind_tools([multiply])

# Invoke the LLM with input that triggers the tool call
msg = llm_with_tools.invoke("100乘200等于多少？")

# Get the tool call
msg.tool_calls
print("工具调用:", msg.tool_calls)
