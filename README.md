\# JD简历匹配智能Agent

基于LangGraph + DeepSeek + Chroma‑RAG实现的简历‑JD匹配Agent。

ReAct智能体，支持PDF简历解析、工具调用、会话持久化，控制台交互，附带Gradio简易WebUI。



\## ✨功能特性

1\. LangGraph实现ReAct Agent，5个自定义工具

&#x20;   - extract\_jd\_requirements：提取JD硬性招聘条件

&#x20;   - parse\_resume：解析简历文本信息

&#x20;   - calc\_match\_score：计算0‑100匹配分数，输出缺失技能

&#x20;   - generate\_suggestion：输出简历、项目改写优化建议

&#x20;   - retrieve\_local\_docs：RAG工具读取本地PDF简历

2\. RAG：Chroma向量库读取PDF简历；无PDF降级手动粘贴文本

3\. 会话持久化：JSON保存完整Agent会话，支持load加载历史对话

4\. 控制台指令：`new`新建任务，`exit`退出，`load sess\_xxx`加载历史会话

5\. 附带Gradio极简网页前端



\## 🛠技术栈

\- Python 3.11

\- Agent编排：LangGraph、LangChain

\- LLM：DeepSeek‑Chat（Function‑Calling）

\- RAG：Chroma、PyPDFLoader

\- WebUI：Gradio



\## 📁项目目录

AI‑Agent/

├─ .env.example          # 环境变量模板，复制为.env 填入自己的 key

├─ requirements.txt      # 依赖清单

├─ main.py               # 控制台主程序

├─ tools.py              # 全部 Agent 工具 + RAG 逻辑

├─ agent\_workflow.py     # LangGraph 工作流定义

├─ gradio\_app.py         # Gradio 网页 Demo

├─ docs/                 # 存放简历 PDF

├─ chroma\_db/            # 向量库自动生成（git 忽略）

└─ sessions/             # 会话 JSON 自动生成（git 忽略）



\## 🚀快速运行

\### 1.环境准备

```bash

conda create -n agent\_demo python=3.11 -y

conda activate agent\_demo

pip install -r requirements.txt



\### 2. 配置密钥



复制 `.env.example` → 新建 `.env`，填入你的 DeepSeek API\_KEY。



\### 3. 控制台运行



```

python main.py

```



> 

> 将简历 PDF 放入 docs 文件夹，程序自动读取 PDF 简历。

> ⚠️修改 PDF 之后，删除 chroma\_db 文件夹重建向量库。



\### 4.Gradio 网页运行



```

python gradio\_app.py

```



浏览器访问 \[http://127.0.0.1:7860](http://127.0.0.1:7860)



\## 💡控制台指令说明



\- `new`：开启新一轮 JD‑简历匹配任务

\- `exit`：退出程序

\- `load sess\_xxxxxx`：加载历史会话文件继续对话



\## 📌遇到的问题与踩坑记录



1\. langchain‑community 包官方停止维护，部分加载器还需要继续使用

2\. Chroma 首次运行会自动下载 onnx 嵌入模型，网络差会超时

3\. PDF 扫描图片版无法提取文字，必须是可复制文字 PDF

4\. LangChain 消息对象不能直接 JSON 序列化，手动实现序列化 / 反序列化

5\. 更新 PDF 必须手动删除 chroma\_db，向量库不会自动检测文件变更



\## 📝后续优化方向



1\. 消息窗口截断，防止 token 无限上涨

2\. reranker 检索重排序优化 RAG 召回效果

3\. 完善 Gradio 网页端全部功能（加载历史会话）

