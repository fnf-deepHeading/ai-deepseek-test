import argparse
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

def process_pdf(pdf_path):
    # PDF 텍스트 로드
    loader = PDFPlumberLoader(pdf_path)
    docs = loader.load()

    # 텍스트를 의미적 청크로 분할
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)

    # 임베딩 생성
    embeddings = HuggingFaceEmbeddings()
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    return retriever

def setup_qa_chain(retriever):
    # 모델 선정
    llm = Ollama(model="deepseek-rl:1.5b")

    # 프롬프트 템플릿 작성
    prompt = """
    1. 아래 맥락만 사용하세요.
    2. 확실하지 않은 경우 "모르겠습니다"라고 말하세요.
    3. 답변은 4문장을 넘지 않도록 하세요.

    맥락: {context}

    질문: {question}

    답변:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    # 체인 1: 답변 생성
    llm_chain = LLMChain(
        llm=llm,
        prompt=QA_CHAIN_PROMPT,
        verbose=True
    )

    # 체인 2: 문서 청크 결합
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="맥락:\ncontent:{page_content}\nsource:{source}"
    )

    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt
    )
    
    return RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever,
        return_source_documents=True
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print("Error: PDF file not found!")
        return
    
    print("Processing PDF...")
    retriever = process_pdf(args.pdf_path)
    qa_chain = setup_qa_chain(retriever)
    
    while True:
        user_input = input("PDF에 대해 질문하세요 (종료하려면 'exit' 입력): ")
        if user_input.lower() == "exit":
            break
        
        response = qa_chain(user_input)["result"]
        print("Response:", response)

if __name__ == "__main__":
    main()
