# test_deepseek.py
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0.1
)

resp = llm.invoke("你好，请简单做个自我介绍")
print("====大模型返回结果====")
print(resp.content)
