import streamlit as st
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
from pypdf import PdfReader

# ==========================================
# 1. 页面配置与资源初始化
# ==========================================
st.set_page_config(page_title="ChatWiki Pro", layout="wide", page_icon="📚")
st.title("📚 ChatWiki Pro: 私域知识库助手")

# 使用缓存避免模型重复加载
@st.cache_resource
def init_engines():
    print("正在初始化语义引擎...")
    # 采用 paraphrase-multilingual 确保中英文支持良好
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    qdrant = QdrantClient(":memory:")
    qdrant.create_collection(
        collection_name="wiki",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    return model, qdrant

embedding_model, qdrant = init_engines()

# 初始化 Session State (对话历史)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed" not in st.session_state:
    st.session_state.processed = False

# ==========================================
# 2. 侧边栏：配置与文件摄取
# ==========================================
with st.sidebar:
    st.header("⚙️ 配置中心")
    api_key = st.text_input("DeepSeek API Key", type="password", help="请填入 sk- 开头的 API 密钥")
    
    st.divider()
    
    st.subheader("📄 文档摄取")
    chunk_size = st.slider("切片大小 (Chunk Size)", 100, 1000, 500)
    uploaded_file = st.file_uploader("上传 PDF 或 TXT 文档", type=["txt", "pdf"])
    
    if st.button("清空对话历史"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# 3. 文档处理逻辑 (ETL Pipeline)
# ==========================================
if uploaded_file and not st.session_state.processed:
    with st.spinner("正在解析文档并构建向量索引..."):
        try:
            content = ""
            # A. 处理 PDF
            if uploaded_file.name.endswith(".pdf"):
                reader = PdfReader(uploaded_file)
                for page in reader.pages:
                    text = page.extract_text()
                    if text: content += text
            # B. 处理 TXT
            else:
                content = uploaded_file.read().decode("utf-8")
            
            if len(content.strip()) < 10:
                st.error("文档内容太少，请检查文件！")
            else:
                # 语义切分
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, 
                    chunk_overlap=int(chunk_size * 0.1)
                )
                chunks = splitter.split_text(content)
                
                # 向量化并入库
                points = []
                for i, text in enumerate(chunks):
                    vector = embedding_model.encode(text).tolist()
                    points.append(PointStruct(id=i, vector=vector, payload={"content": text}))
                
                qdrant.upsert(collection_name="wiki", points=points)
                st.session_state.processed = True
                st.success(f"✅ 解析成功！已切分并索引 {len(chunks)} 个语义片段。")
        except Exception as e:
            st.error(f"解析失败: {e}")

# ==========================================
# 4. 聊天界面渲染
# ==========================================
# 显示历史聊天气泡
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 底部输入框
if prompt := st.chat_input("基于您的文档，问我任何问题..."):
    if not api_key:
        st.warning("请在侧边栏配置 API Key 以启用对话功能。")
        st.stop()
    
    # 1. 渲染用户问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. RAG 回答生成
    with st.chat_message("assistant"):
        with st.spinner("检索资料并思考中..."):
            try:
                # 2.1 语义检索 (Retrieval)
                query_v = embedding_model.encode(prompt).tolist()
                search_results = qdrant.query_points(
                    collection_name="wiki", 
                    query=query_v, 
                    limit=4 # 扩大召回范围
                ).points
                
                # 合并知识块
                context = "\n\n---\n\n".join([hit.payload['content'] for hit in search_results])
                
                # 2.2 构造 DeepSeek 消息流 (Conversation Chain)
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                
                messages = [
                    {"role": "system", "content": "你是一个高效的 AI 知识库助手。请优先基于【参考资料】回答问题。如果资料中未提及，请结合对话历史或诚实说明。回答应精炼、准确。"}
                ]
                # 注入最近 5 轮历史
                messages.extend(st.session_state.messages[-5:])
                # 注入当前 RAG 上下文
                full_prompt = f"【参考资料】：\n{context}\n\n【问题】：{prompt}"
                messages[-1]["content"] = full_prompt # 替换最后一条用户消息为带背景的 Prompt

                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=0.3
                )
                
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"对话出错: {e}")
