from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url="https://api.siliconflow.cn/v1/", model="deepseek-ai/DeepSeek-V3.1-Terminus",
                 api_key="***")

response = llm.invoke("什么是RAG")
print(response.content)

