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

class ChatBot:
    def __init__(self):
        load_dotenv()

        # === Paths & Config ===
        self.persist_directory = "./chroma_db"
        self.embedding_model_name = "sentence-transformers/all-mpnet-base-v2"

        # === Embedding Model ===
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

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

        Structure your response like this:

        1. **Reflection** (max 2 sentences): Share a kind, encouraging comment on the student's question. Was it thoughtful, curious, or well-phrased? Use warm, human-style emojis like 😊, 😄, 🙌, or 😅 to make it feel more personal and relaxed.
        2. **Answer**: Provide a clear and accurate response based on the course material. Use plain language. Bullet points are okay if helpful.
        3. **Follow-up**: Offer a tip, encouragement, or next-step suggestion. If possible, recommend a section title, topic, or keyword from the course material. Feel free to add friendly emojis here too.

        Keep it conversational and approachable — like you're talking to a student one-on-one. Avoid robotic or overly formal language.

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
if __name__ == "__main__":
    chatbot = ChatBot()
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = chatbot.chat(user_input)
        print("Bot:", response)
