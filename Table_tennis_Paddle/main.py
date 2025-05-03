import os
from openai import OpenAI
client = OpenAI(
 api_key=os.environ.get("AI_STUDIO_API_KEY"), # 含有 AI Studio 访问令牌的环境变量，https://aistudio.baidu.com/account/accessToken,
 base_url="https://aistudio.baidu.com/llm/lmapi/v3", # aistudio 大模型 api 服务域名
)

chat_completion = client.chat.completions.create(
 messages=[
 {'role': 'system', 'content': '你是一名专业的乓乓球教练，你精通运动员训练相关知识，负责给运动员提供专业的乒乓球相关帮助建议。'},
 {'role': 'user', 'content': '你好，请介绍一下AI Studio'}
 ],
 model="ernie-4.5-8k-preview",
 stream=True,
)

for chunk in chat_completion:
    if (len(chunk.choices) > 0):
        print(chunk.choices[0].delta.content, end="", flush=True)