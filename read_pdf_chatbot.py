from pathlib import Path
from langchain.document_loaders import DirectoryLoader, PyPDFLoader

# =====================================
# PDFファイルの一括読み込み
# =====================================
pdf_loader = DirectoryLoader(
    "./documents",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)
documents = pdf_loader.load()

# =====================================
# 以降は上記と同じ処理
# =====================================
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="mistral")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

llm = Ollama(model="mistral", temperature=0.7)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

answer = qa.invoke("質問")
print(answer)
