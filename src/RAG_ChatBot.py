import streamlit as st
import chromadb
chromadb.api.client.SharedSystemClient.clear_system_cache()
import os
import glob
from dotenv import load_dotenv
from markitdown import MarkItDown
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

class ChatBot:
    def __init__(self):
        load_dotenv()

        # === Paths & Config ===
        self.persist_directory = "./chroma_db"
        self.embedding_model_name = "sentence-transformers/all-mpnet-base-v2"

        # === Embedding Model ===
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # === Load or Create Vector DB ===
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            vectordb = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=embeddings
            )
        else:
            pdf_files = glob.glob('./materials/*.pdf')
            documents = []
            mid = MarkItDown()
            for file_path in pdf_files:
                result = mid.convert(file_path)
                content = result.text_content
                documents.append(Document(page_content=content, metadata={"source": file_path}))

            text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
            docs = text_splitter.split_documents(documents)
            
            vectordb = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                collection_name="erm_tutor_openai",  # <-- untuk menghapus sampah database
                persist_directory=self.persist_directory
            )
            vectordb.persist()

        # === LLM Setup ===
        llm = ChatOpenAI(
            model_name="gpt-4o",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
            stream=False
        )

        # === Prompt Template ===
        template = """
        You're an AI mentor and tutor for the Empirical Research Methods (ERM) course. When a student asks a question, reply like a supportive human mentor — friendly, encouraging, and natural.

        IMPORTANT RULE: If the student asks a question that is completely outside the context of the provided materials or outside the Empirical Research Methods course, do not try to answer it. Instead, reply strictly with this exact sentence:
        "Sorry, I cannot answer that, because I am ERM Tutor, I only want you to ask questions related to that.". But if the students ask who you are and something related to you, you have to answer it.
        
        Structure your response like this:
        
        Provide a natural, conversational, and cohesive response in paragraphs. 
        You must still include an empathetic reflection, the factual answer based on the context, 
        and an engaging follow-up question, but DO NOT use any explicit labels, headings, 
        or bullet points like 'Reflection:', 'Answer:', or 'Follow-up:'. 
        Weave them seamlessly into a natural human-like reply.
        Keep it conversational and approachable — 
        like you're talking to a student one-on-one. Avoid robotic or overly formal language.

        Context: {context}
        Question: {question}
        Response:
        """



        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        # === RAG Chain ===
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
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

        word_count = len(answer.split())

        print("\n--- Retrieved Chunks ---")
        for i, doc in enumerate(sources):
            print(f"[{i+1}] Source: {doc.metadata.get('source', 'Unknown')}")
            print(doc.page_content[:500], "...\n")

        print(f"🧠 Word count of answer: {word_count}")

        if not sources:
            return "This question is not related to my knowledge."

        return f"{answer}\n\n🧠 (Answer contains {word_count} words)"


# === Example Usage ===
# 1. Definisikan fungsinya tanpa spasi di kiri (sejajar margin)
@st.cache_resource
def inisiasi_tutor_ai():
    return ChatBot()

# 2. Jalankan perintah if
if __name__ == "__main__":
    # 3. Baris ini wajib menjorok ke dalam (tekan Tab 1x)
    chatbot = inisiasi_tutor_ai()
 #   while True:
  #      user_input = input("You: ")
   #     if user_input.lower() in ["exit", "quit"]:
    #        break
     #   response = chatbot.chat(user_input)
      #  print("Bot:", response)
