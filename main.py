# 참고 블로그 : https://apidog.com/kr/blog/rag-deepseek-r1-ollama/
import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

# color palette
primary_color = "#1E90FF"
secondary_color = "#FF6347"
background_color = "#F5F5F5"
text_color = "#4561e9"

# Custom CSS
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stTextInput>div>div>input {{
        border: 2px solid {primary_color};
        border-radius: 5px;
        padding: 10px;
        font-size: 16px;
    }}
    .stFileUploader>div>div>div>button {{
        background-color: {secondary_color};
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    </style>
""", unsafe_allow_html=True)

# Streamlit app title
st.title("Build a RAG System with DeepSeek R1 & Ollama")

# Load the PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # ---- PDF 업로드 및 처리 ---- #
    # streamlit 파일 업로더
    uploaded_file = st.file_uploader("PDF 파일 업로드", type="pdf")

    if uploaded_file:
        # PDF를 임시로 저장
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())

        # PDF 텍스트 로드
        loader = PDFPlumberLoader("temp.pdf")
        docs = loader.load()

    # ---- PDF를 더 작은 의미있는 청크로 나누기 ---- #
    ### 좋은 청크 : 온전한 컨셉/문장, 온전한 Q&A 세트, 논리적인 그룹핑
    ### 나쁜 청크 : 문장 중간을 자른 것, Q&A 가 분리된 것, 문맥이 없는 것
    # 텍스트를 의미적 청크로 분할
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)

    # ---- 검색 가능한 지식 기반 생성 ---- #
    ### 청크에 대한 벡터 임베딩을 생성하고, 이를 FAISS 인덱스에 저장
    ### >> 텍스트를 쿼리하기 쉬운 숫자 형태로 변환하는 과정
    ### >> 나중에 이 인덱스를 기준으로 가장 맥락적으로 관련된 청크를 찾음
    # 임베딩 생성
    embeddings = HuggingFaceEmbeddings()
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # ---- DeepSeek R1 구성 ---- #
    ### RetrievalQA 체인을 인스턴스화... 이게 무슨말이징
    # 모델 선정
    llm = Ollama(model="deepseek-rl:1.5b")

    # 프롬프트 템플릿 작성
    ### 모델이 PDF의 콘텐츠에 답변을 기반하도록 함.
    prompt = """
    1. 아래 맥락만 사용하세요.
    2. 확실하지 않은 경우 "모르겠습니다"라고 말하세요.
    3. 답변은 4문장을 넘지 않도록 하세요.

    맥락: {context}

    질문: {question}

    답변:
    """

    # 언어모델을 FAISS 인덱스와 연결된 검색기로 감싸면서 체인에 수행된 모든 쿼리는 PDF의 내용을 참조하여 답변을 제공
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    # ---- RAG 체인 조립 ----
    ### 업로드 > 청킹 > 검색
    # 체인 1: 답변 생성
    llm_chain = LLMChain(
        llm=llm,
        prompt=QA_CHAIN_PROMPT,
        callbacks=None,
        verbose=True
    )

    # 체인 2: 문서 청크 결합
    
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="맥락:\ncontent:{page_content}\nsource:{source}",
    )

    # 최종 RAG 파이프라인
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        callbacks=None
    )
    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        verbose=True,
        retriever=retriever,
        return_source_documents=True
    )

    # User input
    user_input = st.text_input("PDF에 대해 질문하세요:")

    # Process user input
    if user_input:
        with st.spinner("고민중..."):
            response = qa(user_input)["result"]
            st.write("Response:")
            st.write(response)
else:
    st.write("Please upload a PDF file to proceed.")