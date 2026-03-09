## RAG_construction_using_local_LLM
Ollamaを使ったフリーのRAG構築方法

基本的なチャットbot構築後モデルに読み込む学習データの読み込ましを行う


## 必要な環境構築
用意するもの

・Ollamaのインストール 

・モデルの確認

・pythonライブラリ



#### Ollamaのインストール

Ollama公式サイト からダウンロード　https://ollama.com/

Windows / Mac / Linux に対応

インストール後、ターミナルで確認：

ollama --version



#### モデルの確認

##### 軽量な Mistral（推奨）

ollama pull mistral


##### 日本語対応の Llama 2

ollama pull llama2



##### より高精度の Neural Chat

ollama pull neural-chat


##### 最新の Llama 2 日本語版

ollama pull elyza/elyza-jp-8b


Ollamaサーバーの起動

ollama serve



#### 必要なpythonライブラリ

pip install langchain langchain-community chromadb ollama



 ## Ollamaを使ったRAG実装
 chatBot.py：ollamaで導入したモデルを動かすチャットボットプログラム
 
 read_pdf_chatbot.py:PDF等の資料を読み込んで実装するチャットボットプログラム

 
 

