from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import os

# =====================================
# 1. ドキュメント読み込み
# =====================================
loader = TextLoader("sample.txt", encoding="utf-8")
documents = loader.load()

# =====================================
# 2. テキスト分割
# =====================================
splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

# =====================================
# 3. 埋め込みモデルの設定（Ollamaで実行）
# =====================================
embeddings = OllamaEmbeddings(
    model="mistral",
    base_url="http://localhost:11434"
)

# =====================================
# 4. ベクトルデータベース作成・保存
# =====================================
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# =====================================
# 5. LLMの設定（Ollama）
# =====================================
llm = Ollama(
    model="mistral",
    base_url="http://localhost:11434",
    temperature=0.7
)

# =====================================
# 6. RAGチェーン構築
# =====================================
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# =====================================
# 7. 質問実行
# =====================================
question = "あなたの質問をここに入力"
answer = qa.invoke(question)
print("回答:", answer)
