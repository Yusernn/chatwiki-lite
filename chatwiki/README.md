# ChatWiki-Lite: 基于 RAG 架构的轻量级知识库对话系统

这是一个基于检索增强生成（RAG）技术实现的本地知识库问答工具。

## 核心功能
- **多格式支持**：支持 PDF 与 TXT 文档的实时解析与摄取。
- **语义搜索**：基于 Sentence-Transformers 实现高维向量检索。
- **检索溯源**：内置“调试模式”，可实时查看 AI 回答的原始参考片段及相似度得分。
- **对话记忆**：支持多轮对话上下文关联。

## 技术栈
- **LLM**: DeepSeek-V3 (via API)
- **Vector DB**: Qdrant (In-Memory)
- **Embedding**: MiniLM-L12-v2 (Local)
- **UI**: Streamlit

## 快速启动
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
2. 启动应用：
```Bash
  streamlit run app.py





