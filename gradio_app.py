# gradio_app.py 极简web骨架
import gradio as gr
from agent_workflow import graph
from tools import retrieve_local_docs, build_vector_store
import os
from datetime import datetime
# 把main.py里面序列化相关函数全部复制到此文件
from langchain_core.messages import HumanMessage, SystemMessage

def gen_session_id():
    return datetime.now().strftime("sess_%Y%m%d_%H%M%S")

def analyse(jd_text, pdf_file, state):
    """点击开始分析触发"""
    session_id = gen_session_id()
    messages = [
        SystemMessage(content="你是简历匹配Agent。必须完整走完工具调用流程...")
    ]
    # 如果上传PDF，保存并重建向量库
    if pdf_file is not None:
        # 复制pdf到docs，然后重建向量库（先删除旧chroma_db）
        import shutil
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        shutil.copy(pdf_file, "./docs/upload_resume.pdf")
        build_vector_store()
        resume_content = retrieve_local_docs.invoke({"query":"提取全部简历内容"})
    else:
        return state,"请上传PDF简历"

    user_input = f"岗位JD：{jd_text}\n候选人简历：{resume_content}"
    messages.append(HumanMessage(content=user_input))
    result = graph.invoke({"messages":messages})
    out_text = result["messages"][-1].content
    # 保存会话
    state["session_id"] = session_id
    state["messages"] = result["messages"]
    return state, out_text

def chat(message, state):
    """聊天追问"""
    if not state.get("messages"):
        return "请先完成JD简历分析！", state
    messages = state["messages"]
    messages.append(HumanMessage(content=message))
    res = graph.invoke({"messages":messages})
    state["messages"] = res["messages"]
    return res["messages"][-1].content, state

with gr.Blocks(title="JD简历匹配Agent") as demo:
    state = gr.State({}) # 存储session_id、messages
    gr.Markdown("# JD‑简历匹配智能Agent")
    with gr.Row():
        jd_input = gr.Textbox(label="粘贴岗位JD", lines=8)
        pdf_upload = gr.File(label="上传简历PDF", file_types=[".pdf"])
    submit_btn = gr.Button("开始匹配分析")
    output = gr.Markdown(label="分析报告")
    gr.Markdown("## 继续对话追问")
    chatbot_input = gr.Textbox(label="输入问题")
    chat_output = gr.Markdown()
    chat_btn = gr.Button("发送")

    submit_btn.click(analyse, inputs=[jd_input,pdf_upload,state], outputs=[state, output])
    chat_btn.click(chat, inputs=[chatbot_input, state], outputs=[chat_output, state])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
