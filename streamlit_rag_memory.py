import os
import tempfile
import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# PDF 파일 로드 함수
def load_pdf(_file):
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
        tmp_file.write(_file.getvalue())
        tmp_file_path = tmp_file.name
        loader = PyPDFLoader(file_path=tmp_file_path)
        pages = loader.load_and_split()
    return pages

# 텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(_docs)
    persist_directory = "./chroma_db"
    vectorstore = Chroma.from_documents(
        split_docs,
        OpenAIEmbeddings(),
        persist_directory=persist_directory
    )
    return vectorstore

# Streamlit UI
st.header("헌법 Q&A 챗봇 💬 📚")

# GPT 모델 선택
option = st.selectbox("Select GPT Model", ["gpt-4", "gpt-3.5-turbo"])

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])
if not uploaded_file:
    st.warning("PDF 파일을 업로드해주세요.")
    st.stop()

# PDF 처리
pages = load_pdf(uploaded_file)

# Chain 생성
@st.cache_resource
def chaining(_pages, selected_model):
    vectorstore = create_vector_store(_pages)
    retriever = vectorstore.as_retriever()

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer perfect. please use imogi with the answer.
    대답은 한국어로 하고, 존댓말을 써줘.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model=selected_model)
    return llm  # 단순화

rag_chain = chaining(pages, option)

# 대화 기록 관리
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "헌법에 대해 무엇이든 물어보세요!"}]

# 기존 메시지 출력
for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# 사용자 입력 처리
if prompt_message := st.chat_input("Your question"):
    st.session_state["messages"].append({"role": "user", "content": prompt_message})
    st.chat_message("human").write(prompt_message)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain({"input": prompt_message})
            st.session_state["messages"].append({"role": "assistant", "content": response["content"]})
            st.write(response["content"])
