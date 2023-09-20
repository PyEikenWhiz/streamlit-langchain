from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import streamlit as st

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3

long_text = """
GPT-4は、OpenAIが開発したAI技術であるGPTシリーズの第4世代目のモデルです。

自然言語処理(NLP)という技術を使い、文章の生成や理解を行うことができます。

これにより、人間と同じような文章を作成することが可能です。

GPT-4は、トランスフォーマーアーキテクチャに基づいており、より強力な性能を発揮します。

GPT-4は、インターネット上の大量のテキストデータを学習し、豊富な知識を持っています。

しかし、2021年9月までの情報しか持っていません。

このモデルは、質問応答や文章生成、文章要約など、様々なタスクで使用できます。

ただし、GPT-4は完璧ではありません。

時々、誤った情報や不適切な内容を生成することがあります。

使用者は、その限界を理解し、

適切な方法で利用することが重要です。
"""
print(len(long_text))
with open("./long_text.txt", "w") as f:
    f.write(long_text)
    f.close()

loader = TextLoader('./long_text.txt')

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

from langchain.embeddings import HuggingFaceEmbeddings #sentence-transformers
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import HuggingFacePipeline

# Function to load the language model
@st.cache_resource  # This decorator caches the result for better performance
def load_language_model():
    model_id = "rinna/japanese-gpt2-small"
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.do_lower_case = True  # due to some bug of tokenizer config loading
    model_ = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model_, tokenizer=tokenizer, max_new_tokens=64)
    return HuggingFacePipeline(pipeline=pipe)

model = load_language_model()

docsearch = Chroma.from_documents(texts, embeddings)
qa = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=docsearch.as_retriever())

query = "GPT-4とは？"
st.write(query)
print(qa.run(query))
st.write(qa.run(query))
