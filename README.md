# 🎓 ERM Tutor-AI: A RAG-Based Educational Tutor

This project was initially completed and presented in November 2025, during which the system was running exclusively in a local environment. In October 2026, the source code and database were published to GitHub and deployed via Streamlit Cloud, allowing its functionality to be accessed and tested directly online.

Developed as part of a thesis project for the Master of Educational Technology program at Saarland University, this project integrates Generative Artificial Intelligence (GAI) with the Retrieval-Augmented Generation (RAG) framework to create an interactive AI tutor assistant for students taking the Empirical Research Methods (ERM) course.

---

## 📖 Background and Problem Statement

Although Large Language Models (LLMs) possess remarkable generative capabilities, they are prone to "hallucinations"—a phenomenon where the AI generates information that sounds plausible but is factually incorrect or unfounded. In a higher education context that demands strict precision, these inaccuracies can mislead students and disrupt the learning process. 

To mitigate this risk, ERM Tutor-AI combines the capabilities of a GPT-based model with a curated knowledge database containing validated ERM course materials and literature[cite: 11]. By leveraging the RAG architecture, the system is forced to retrieve information directly from official course materials before generating a response, ensuring that the provided answers are always accurate, contextually relevant, and free from misinformation.

---

## 🛠️ Technology Stack

*   **Frontend / UI:** Streamlit[cite: 57] for building an interactive, user-friendly web interface that facilitates real-time dialogue[cite: 61, 62].
*   **Large Language Model (LLM):** OpenAI GPT-4o accessed via API for advanced natural language understanding and generation[cite: 57, 58].
*   **Orchestration Framework:** LangChain to manage the data flow between the user interface, the vector database, and the LLM[cite: 60].
*   **Vector Database:** ChromaDB[cite: 57] for storing embedded text chunks and performing efficient semantic similarity searches[cite: 60].
*   **Embeddings:** `sentence-transformers` (via Hugging Face) to convert text chunks into high-dimensional numeric representations to capture semantic meaning[cite: 58].
*   **Document Parsing:** Markitdown for extracting structured metadata from course PDFs to ensure high searchability and transparency[cite: 60].

---

## 📂 Project Structure

```text
erm_tutorai/
│
├── chroma_db/                 # Pre-built vector database containing ERM course embeddings
├── src/
│   ├── RAG_ChatBot.py         # Backend logic, LLM integration, and RAG pipeline
│   └── streamlit.py           # Frontend UI, chat interface, and session state management
│
├── logo_ermai.png             # Application logo and visual assets
├── requirements.txt           # List of project dependencies
└── README.md                  # Project overview and instructions

---

📦 Dependencies

The project relies on the following primary Python libraries (specified in requirements.txt):
streamlit
langchain
langchain-openai
langchain-community
chromadb
sentence-transformers
pillow

📊 Evaluation and Performance Results

The reliability of the ERM Tutor-AI system was comprehensively tested through a direct performance comparison against a baseline LLM model (ChatGPT-4o). Both systems were given 20 representative questions drawn from the ERM curriculum[cite: 11]. A panel of instructors evaluated these responses based on strict criteria, including factual accuracy, relevance, clarity, and tutoring behavior.

1. Superior Accuracy and Relevance: ERM Tutor-AI consistently outperformed ChatGPT-4o, achieving an average quality score of 6.75 out of a 7.00 scale, compared to the 6.00 score achieved by the baseline model.
2. Context-Based Precision: The evaluation results proved that ERM Tutor-AI is capable of delivering answers that are instructionally structured and contextually sound.
3. Pedagogical Tutoring Behavior: Unlike standard LLMs that tend to provide lengthy but unfocused answers, ERM Tutor-AI is designed to act like a mentor. The system provides an appropriate amount of information, anticipates user confusion, and actively guides students using structured reflections, feedback, and follow-up questions.

 © 2026 Amrina. All rights reserved.
