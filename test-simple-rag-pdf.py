import argparse
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
# from langchain.schema.runnable import RunnableLambda
# from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap
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
    llm = OllamaLLM(model="deepseek-r1:1.5b")

    # 프롬프트 템플릿 작성
    # 프롬프트들은 json 으로 저장 후 로드 가능 ( 책 p.61 참고 )
    prompt = """
    1. 아래 맥락만 사용하세요.
    2. 확실하지 않은 경우 "모르겠습니다"라고 말하세요.
    3. 답변은 4문장을 넘지 않도록 하세요.

    맥락: {context}

    질문: {question}

    답변:
    """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt) # PromptTemplate 초기화 과정 ( input_variables 지정 안해도 됨 )

    # 체인 1: 답변 생성
    # llm_chain = QA_CHAIN_PROMPT | llm

    # 체인 2: 문서 청크 결합
    # document_prompt = PromptTemplate(
    #     input_variables=["page_content", "source"],
    #     template="맥락:\ncontent:{page_content}\nsource:{source}"
    # )

    # combine_documents_chain = StuffDocumentsChain(
    #     llm_chain=llm_chain,
    #     document_variable_name="context",
    #     document_prompt=document_prompt
    # )
    
    # return RetrievalQA(
    #     combine_documents_chain=combine_documents_chain,
    #     retriever=retriever,
    #     return_source_documents=True
    # )

    qa_chain = (
        RunnableMap({"context": retriever})  # 문서 검색
        | QA_CHAIN_PROMPT  # 프롬프트 적용
        | llm  # LLM 실행
        | StrOutputParser()  # 결과 파싱
    )

    return qa_chain

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
        
        # response = qa_chain(user_input)["result"]
        # 🔹 검색을 위해 retriever에 `user_input`만 전달해야 함
        relevant_docs = retriever.invoke(user_input)

        # 🔹 검색된 문서들을 하나의 context로 합침
        context = "\n".join([doc.page_content for doc in relevant_docs])

        # 🔹 이제 LLM에 `context`와 `question`을 함께 전달
        response = qa_chain.invoke({"context": context, "question": user_input})
        print("Response:", response)

if __name__ == "__main__":
    main()
