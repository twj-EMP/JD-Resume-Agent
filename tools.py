# tools.py
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

load_dotenv()
llm = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("BASE_URL"),
    model=os.getenv("MODEL_NAME"),
    temperature=0.1
)

# 向量库持久化路径
CHROMA_PATH = "./chroma_db"
DOC_FOLDER = "./docs"


def build_vector_store():
    """加载docs下全部PDF，切分文本，构建向量库；PDF文件更新时调用一次"""
    all_docs = []
    if not os.path.exists(DOC_FOLDER):
        os.makedirs(DOC_FOLDER)

    for fname in os.listdir(DOC_FOLDER):
        if fname.lower().endswith(".pdf"):
            pdf_path = os.path.join(DOC_FOLDER, fname)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            all_docs.extend(pages)

    if not all_docs:
        return None

    # 文本切分
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    split_docs = splitter.split_documents(all_docs)

    db = Chroma.from_documents(
        documents=split_docs,
        persist_directory=CHROMA_PATH
    )
    return db


def get_vector_store():
    """获取向量库实例，如果磁盘已有就直接加载，没有就构建"""
    if os.path.exists(CHROMA_PATH) and len(os.listdir(CHROMA_PATH)) > 0:
        db = Chroma(persist_directory=CHROMA_PATH)
        return db
    return build_vector_store()


@tool
def retrieve_local_docs(query: str) -> str:
    """
    检索本地docs文件夹中的PDF文档（简历、项目文档）。
    当需要获取PDF简历内容、本地项目资料的时候调用该工具。
    参数 query：检索查询词，例如“简历项目经历”、“个人技能”
    返回：文档的相关片段文本
    """
    db = get_vector_store()
    if db is None:
        return "本地没有找到PDF文档，请手动粘贴简历文本。"
    # 检索top3片段
    docs = db.similarity_search(query, k=3)
    content_list = [d.page_content for d in docs]
    return "\n---\n".join(content_list)


@tool
def extract_jd_requirements(jd_text: str) -> str:
    """
    从岗位JD文本提取岗位硬性要求：技术栈、工作年限、学历
    参数 jd_text：原始JD岗位描述文本
    返回：结构化的岗位要求清单字符串
    """
    prompt = f"""
请仔细提取下面岗位JD里面全部硬性招聘条件：需要掌握的技术栈、要求工作年限、学历要求。
输出清晰结构化清单，不要多余闲聊。
JD内容：
{jd_text}
"""
    res = llm.invoke(prompt)
    return res.content


@tool
def parse_resume(resume_text: str) -> str:
    """
    解析简历，提取候选人掌握技能、项目经历、相关经验
    参数 resume_text：原始简历文本
    返回：简历提取后的结构化信息
    """
    prompt = f"""
提取这份简历中的：掌握的技术、做过的项目、相关开发经验，输出结构化清单。
简历：
{resume_text}
"""
    res = llm.invoke(prompt)
    return res.content


@tool
def calc_match_score(jd_require: str, resume_skill: str) -> str:
    """
    根据JD要求与简历信息，计算0‑100匹配分数，列出缺失技能
    参数 jd_require：extract_jd_requirements输出的岗位要求
    参数 resume_skill：parse_resume输出的简历信息
    返回：匹配分数、缺失技能清单
    """
    prompt = f"""
对比下面岗位招聘要求和候选人简历信息：
1.给出0‑100的匹配分数
2.清晰列出简历里面缺失哪些岗位需要的技能点

岗位要求：
{jd_require}

候选人简历信息：
{resume_skill}

输出格式示例：
匹配分数：XX
缺失技能：
‑ xxx
‑ xxx
"""
    res = llm.invoke(prompt)
    return res.content


@tool
def generate_suggestion(jd_require: str, resume_skill: str, score_info: str) -> str:
    """
    根据匹配结果生成简历优化建议，针对JD做项目描述优化
    参数 jd_require：岗位要求
    参数 resume_skill：简历解析结果
    参数 score_info：calc_match_score返回的分数与缺失技能
    返回：简历修改、项目描述优化建议
    """
    prompt = f"""
结合岗位要求、候选人简历、匹配得分，输出针对性简历优化建议：
1.哪些项目经历可以重点放大
2.缺失技能如何在简历中弥补
3.项目描述怎么改写更贴合这个JD

岗位要求：
{jd_require}
简历信息：
{resume_skill}
匹配得分与缺失项：
{score_info}
"""
    res = llm.invoke(prompt)
    return res.content


# 工具列表，新增 retrieve_local_docs
tools = [extract_jd_requirements, parse_resume, calc_match_score, generate_suggestion, retrieve_local_docs]

if __name__ == "__main__":
    # 单元测试RAG，把简历pdf放到docs下面，运行 python tools.py 测试检索
    db = get_vector_store()
    if db:
        out = retrieve_local_docs.invoke({"query": "项目经历"})
        print(out)
    else:
        print("docs文件夹没有PDF文件")
