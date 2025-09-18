import os
from langchain.chat_models import init_chat_model

if not os.getenv("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = "***"

model = init_chat_model("deepseek:deepseek-chat")
response = model.invoke("Why do parrots talk?")

print(response)
