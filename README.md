# Bengali Textbook RAG Chatbot

A conversational AI chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions about the "HSC Bangla 1st Paper" textbook. The application is built with FastAPI, LangChain, and Google's Gemini Pro model.

## Screenshots

Make sure your screenshot files are named `screenshot1.png` and `screenshot2.png` and are placed inside the `SS` folder.

| Chat Interface | Another View |
| :---: | :---: |
| ![Chat Interface](./SS/Image1.png) | ![Another View](./SS/Image2.png) |

## 📖 About The Project

This project provides an interactive chat interface allowing users to ask specific questions about a Bengali textbook. Instead of relying on pre-trained knowledge, it uses the RAG technique to retrieve relevant passages directly from the source PDF, which are then used by a Large Language Model (LLM) to generate accurate, context-aware answers.

This approach ensures that the chatbot's knowledge is strictly limited to the provided document, making it a reliable tool for study and reference.

### Key Features

* **Interactive Chat UI:** A clean, modern front-end built with HTML and vanilla JavaScript.
* **PDF Text Extraction:** Automatically processes the source PDF using Tesseract OCR on the first run.
* **Advanced Retrieval:** Implements LangChain's `ParentDocumentRetriever` and `MultiQueryRetriever` for more accurate context fetching.
* **Conversational Memory:** Remembers the last few turns of the conversation to answer follow-up questions.
* **Bilingual Support:** Can understand questions and provide answers in both Bengali and English.

### 🛠️ Tech Stack

* **Backend:** Python, FastAPI
* **AI/ML:** LangChain, Google Gemini, HuggingFace Embeddings
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **OCR & PDF:** Tesseract, PyMuPDF
* **Frontend:** HTML, CSS, JavaScript

### Prerequisites

Please install these in your system:


* **Tesseract-OCR Engine**
    * This is a system-level dependency, not a Python package. Your Python code depends on it to perform OCR.
    * [Tesseract Installation Guide]([https://tesseract-ocr.github.io/tessdoc/Installation.html](https://github.com/tesseract-ocr/tesseract.git))
    * On Windows, make sure to note the installation path. The code currently points to `C:\Program Files\Tesseract-OCR\tesseract.exe`. If your path is different, update the `TESSERACT_CMD_PATH` variable in `main.py`.

### Installation and Setup

1.  **Clone the Repository**
    ```sh
    git clone https://github.com/KZ1R2N/RAG-BOT.git
    
    ```

2.  **Create and Activate a Virtual Environment**
    * On Windows:
        ```sh
        python -m venv venv
        venv/scripts/activate  - Powershell
        ```
    * On macOS/Linux:
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install Python Dependencies**
    ```sh
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    * Create a file named `.env` in the root of the project.
    * Add your Google API key to this file:
        ```
        GOOGLE_API_KEY="your_google_api_key_here"
        ```

5.  **Add the Source Document**
    * Place your PDF file, `HSC26-Bangla1st-Paper.pdf`, in the root directory of the project. Already added in the repository. 

##  Usage

1.  **Run the FastAPI Server**
    ```sh
    uvicorn main:app --reload
    ```
    * **First Run Note:** The very first time you start the server, it will begin the OCR process to read the PDF and create `extracted_output.txt`. This is a one-time setup and may take several minutes depending on your computer's speed and the PDF's length. You can monitor the progress in your terminal.

2.  **Open the Chatbot**
    * Once the server is running and the message "🚀 RAG Pipeline with Multi-Query is Ready!" appears, open your web browser and navigate to:
    * [http://127.0.0.1:8000](http://127.0.0.1:8000)

You can now start asking questions about the textbook!



Here I've another project for reference which I had worked on long time ago. Here you can upload any pdf and based on that pdf the chat bot will answer. But that was not made perticularly Bangla in head. 

https://github.com/KZ1R2N/Chat-Pdf.git


Please feel free to contact me if any issue occures regarding the project. 

#Answers: 

I have tried various test extraction technique such as pypdf, pdfplumber, pymupdf. pdfplumber works well on tables as our book got complex structure with tables, MCQ and paragraphs. But even it was struggeling to answwering most of the questions. Then tried to do with unstructured pdf but due to specification limitaion it wasn't supported on my PC. 
Mainly it was problem of the font style. even copy paste in this pdf scrumbled the copied text. If the font was Unicode then it would have worked well with pdfplumber. That's why to extract text I used OCR. For this I have used EasyOCR but it can't recognize all the letter well but pytessaract extracted all the text from the pdf by converting it to high level screen shot and then OCR on it to extract the exact text. 
I have use parent document retriever. Which have two type of chunks: 
Child Chunk : A RecursiveCharacterTextSplitter creates small chunks of text (400 characters). These small, focused chunks are embedded and placed into the FAISS vector store. Their main purpose is to be highly effective at matching the specific semantic meaning of a user's query. 
Parent Chunk: Another RecursiveCharacterTextSplitter defines larger chunks (4000 characters). The smaller "child" chunks are derived from these larger "parent" documents. These parent chunks are kept in a simple document store.
The retrieval process works in two steps: first, it finds the most relevant small child chunks in the vector store, and then it retrieves the corresponding large parent chunks to send to the language model.

This strategy is highly effective because it leverages the strengths of both small and large chunks, creating a "best of both worlds" scenario:

Precise Retrieval: Small chunks are better for semantic search. Their meaning is concentrated, making it easier for the vector search to find a precise match for a user's query without the "noise" from surrounding, unrelated text.

Rich Context: Language models generate better answers when they have more context. By providing the larger parent chunk, you ensure the model sees the full paragraph or section where the relevant information was found. This prevents the model from giving fragmented answers based on incomplete snippets and helps it produce more coherent and accurate responses.

The comparison process happens in two main steps:

Vector Conversion: I use a Hugging Face embedding model called sentence-transformers/paraphrase-multilingual-mpnet-base-v2. This model's job is to read a piece of text (either the user's query or a document chunk) and convert it into a high-dimensional numerical vector. The key is that semantically similar texts will result in vectors that are close to each other in vector space.

Similarity Search: The FAISS (Facebook AI Similarity Search) library takes the query's vector and efficiently searches through all the stored document chunk vectors. It identifies the chunks whose vectors have the highest cosine similarity to the query vector. In simple terms, it finds the chunks that are closest in meaning to the question being asked.

I chose this specific embedding model and storage setup for a few key reasons:

Multilingual Support is Essential: The source document is in Bengali, but users might ask questions in English or a mix of both. The paraphrase-multilingual-mpnet-base-v2 model is specifically trained to handle multiple languages, making it perfect for accurately matching an English query to a Bengali text chunk.

Speed and Efficiency: FAISS is incredibly fast and memory-efficient. It's designed for rapid similarity searches, even with millions of vectors. For this project, it means the retrieval step is nearly instantaneous and can run locally on a standard CPU (faiss-cpu) without requiring a powerful server or GPU.

High-Quality Semantic Matching: My goal wasn't just to find keywords but to match based on the meaning or intent of the question. Sentence-transformer models are excellent at this, and FAISS is the industry standard for performing the search on the resulting vectors. It's a robust and proven combination for building effective RAG systems.

Query Transformation with MultiQueryRetriever: This is my primary strategy. Instead of taking the user's question at face value, I use the LLM to rewrite it into several different, more specific queries from various perspectives. For example, if a user asks "What about the main character?", the retriever might generate variants like "What are the main character's personality traits?" and "What is the main character's role in the story?". It then searches the document for all of these variants, giving me a much richer set of results.

Semantic Embedding Model: The foundation of the comparison is the paraphrase-multilingual-mpnet-base-v2 model. It converts text into vectors based on meaning, not just keywords. This ensures that even if the user's wording is completely different from the book's, as long as the underlying meaning is similar, it will find a match.

CONTEXT ParentDocumentRetriever: After finding the best small, specific chunks of text (the "child" documents), the retriever fetches the larger "parent" chunk they belong to. This guarantees that the LLM receives a full paragraph for context, not just an isolated sentence, which is crucial for a meaningful interpretation.

Yes the results are accurate. It can asnwer maximum questions correctly. But a better paid Model with good context window, increased limit it would work more efficient. ALso if the pdf was unicoded then by using pdfplumber alone would have bring an amazing results. 
