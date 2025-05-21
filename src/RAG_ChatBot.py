from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import glob
from chromadb.config import Settings

class ChatBot:
    def __init__(self):
        load_dotenv()

        # Step 1: Load PDFs
        pdf_files = glob.glob('./materials/*.pdf')
        documents = []
        for file_path in pdf_files:
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        # Step 2: Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        # Step 3: Embedding model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # Step 4: Create in-memory Chroma DB
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
        )

        # Step 5: Initialize GPT-4o (or fallback)
        model_name = "gpt-4o"
        llm = ChatOpenAI(model_name=model_name, openai_api_key=os.getenv("OPENAI_API_KEY"))

        # Step 6: Prompt Template
        template = """
    
        You are an AI tutor of Empirical Research Methos course (ERM). Users will ask you questions about that course. Use the following piece of context to answer the question.
        If you don't know the answer, just say you don't know.
        Your answer should be short, clear and concise, no longer than 3 sentences. you can make a list if the answer including some points.

        Context: {context}
        Question: {question}
        Answer:
        """
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # Step 7: Retriever + QA Chain
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})
        self.rag_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    def chat(self, question: str) -> str:
        result = self.rag_chain(question)
        answer = result["result"]
        sources = result.get("source_documents", [])

        print("\n--- Retrieved Chunks ---")
        for i, doc in enumerate(sources):
            print(f"[{i+1}] {doc.page_content[:500]}...\n")

        if not sources:
            return "This question is not related to my knowledge."

        return answer


# Usage example:
if __name__ == "__main__":
    chatbot = ChatBot()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = chatbot.chat(user_input)
        print("Bot:", response)
