# main.py 完整最终版
from agent_workflow import graph
from tools import retrieve_local_docs
import json
import os
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

SESSION_DIR = "./sessions"
if not os.path.exists(SESSION_DIR):
    os.makedirs(SESSION_DIR)


def serialize_messages(messages):
    """LangChain消息列表 -> 可JSON序列化字典列表"""
    out = []
    for msg in messages:
        d = {"type": msg.type, "content": msg.content}
        if hasattr(msg, "tool_call_id"):
            d["tool_call_id"] = msg.tool_call_id
        if hasattr(msg, "tool_calls"):
            d["tool_calls"] = msg.tool_calls
        out.append(d)
    return out


def deserialize_messages(dicts):
    """字典列表 -> 还原LangChain消息对象列表"""
    msg_list = []
    for d in dicts:
        t = d["type"]
        content = d["content"]
        if t == "system":
            msg = SystemMessage(content=content)
        elif t == "human":
            msg = HumanMessage(content=content)
        elif t == "ai":
            msg = AIMessage(content=content, tool_calls=d.get("tool_calls"))
        elif t == "tool":
            msg = ToolMessage(content=content, tool_call_id=d.get("tool_call_id"))
        else:
            continue
        msg_list.append(msg)
    return msg_list


def save_session(session_id, messages):
    """保存会话到JSON文件"""
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    data = {
        "session_id": session_id,
        "save_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": serialize_messages(messages)
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_session(session_id):
    """加载会话，不存在返回None"""
    path = os.path.join(SESSION_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return deserialize_messages(data["messages"])


def gen_session_id():
    """生成时间戳会话ID"""
    return datetime.now().strftime("sess_%Y%m%d_%H%M%S")


def main():
    print("==== JD‑简历匹配智能Agent ====")
    print("提示：docs文件夹放入简历PDF，程序优先读取PDF简历；")
    print("指令说明：")
    print("  new        → 开启新JD简历任务")
    print("  exit       → 退出程序")
    print("  load sess_xxx → 加载历史会话继续对话\n")

    while True:
        session_id = gen_session_id()
        print(f"\n>>> 新建会话：{session_id}")

        jd_text = input("请粘贴岗位JD内容：\n").strip()
        if not jd_text:
            print("错误：JD不能为空！")
            continue

        # 优先尝试读取本地PDF简历
        pdf_resume = retrieve_local_docs.invoke({"query": "提取全部简历内容：个人信息、技能、项目经历、荣誉"})
        # 判断：无PDF提示 或者 返回内容为空
        if "本地没有找到PDF文档" in pdf_resume or len(pdf_resume.strip()) < 20:
            resume_text = input("\ndocs未检测到PDF / PDF读取内容为空，请粘贴简历文本：\n").strip()
            if not resume_text:
                print("简历不能为空！")
                continue
        else:
            print("\n已读取本地PDF简历！")
            resume_text = pdf_resume


        messages = [
            SystemMessage(content=
                "你是简历匹配Agent。"
                "可以调用retrieve_local_docs读取本地PDF简历文档。"
                "必须完整走完工具调用流程："
                "第一步调用extract_jd_requirements提取JD要求；"
                "第二步调用parse_resume解析简历；"
                "第三步调用calc_match_score计算匹配分；"
                "第四步调用generate_suggestion输出简历优化建议。"
                "不要跳过工具直接回答。")
        ]

        user_input = f"岗位JD：{jd_text}\n候选人简历：{resume_text}"
        messages.append(HumanMessage(content=user_input))

        result = graph.invoke({"messages": messages})
        messages = result["messages"]

        save_session(session_id, messages)
        print("\n======== Agent最终输出 ========")
        print(messages[-1].content)

        # 多轮追问循环
        while True:
            print("\n----------------------------------")
            user_msg = input("\n继续提问(new=新任务，exit=退出，load sess_xxx加载历史):\n")
            user_msg = user_msg.strip()

            if user_msg.lower() == "exit":
                print("程序退出")
                return
            if user_msg.lower() == "new":
                print("\n>>> 切换新任务，重新输入JD\n")
                break
            if user_msg.startswith("load "):
                load_id = user_msg.replace("load ", "").strip()
                loaded_msg = load_session(load_id)
                if loaded_msg:
                    session_id = load_id
                    messages = loaded_msg
                    print(f"成功加载会话 {load_id}")
                    print("\n==== Agent历史最后回复 ====")
                    print(messages[-1].content)
                else:
                    print(f"找不到会话文件 {load_id}.json")
                continue
            if not user_msg:
                print("输入不能为空！")
                continue

            messages.append(HumanMessage(content=user_msg))
            result = graph.invoke({"messages": messages})
            messages = result["messages"]
            save_session(session_id, messages)
            print("\n==== Agent回复 ====")
            print(messages[-1].content)


if __name__ == "__main__":
    main()
