import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import fitz 
import pytesseract
from PIL import Image
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ParentDocumentRetriever



app = FastAPI()
templates = Jinja2Templates(directory="templates")

PDF_PATH = "HSC26-Bangla1st-Paper.pdf"
OCR_TEXT_PATH = "extracted_output.txt"  # The pre-processed text file
TESSERACT_CMD_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

rag_chain = None
memory = None
STARTUP_ERROR_MESSAGE = None

def load_text_from_file(file_path):
    """
    Loads text from the pre-processed TXT file and splits it back into
    LangChain Document objects, one for each page.
    """
    print(f"--- Loading text from {file_path} ---")
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read()
    
    pages = re.split(r'--- Page \d+ ---', full_text)[1:]

    for i, page_content in enumerate(pages, 1):
        if page_content.strip():
            documents.append(Document(page_content=page_content.strip(), metadata={"page": i}))
    
    print(f"--- Loaded {len(documents)} pages from the text file. ---")
    return documents

def ocr_pdf_with_tesseract(pdf_path, output_path):
    """
    One-time function to perform OCR on the PDF using Tesseract and save the result.
    tesseract_config = f'--tessdata-dir "{TESSDATA_DIR}"'
    """
    
    doc = fitz.open(pdf_path)
    full_ocr_text = ""
    print(f"--- Starting Tesseract OCR process for {len(doc)} pages... This will take time. ---")

    for page_num, page in enumerate(doc, 1):
        print(f"  -> OCR on page {page_num}/{len(doc)}...")
        pix = page.get_pixmap(dpi=300)
        
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        page_text = pytesseract.image_to_string(image, lang='Bengali')
        
        full_ocr_text += f"--- Page {page_num} ---\n{page_text}\n\n"
    
    doc.close()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_ocr_text)
    print(f"--- ✅ Tesseract OCR complete. Text saved to {output_path} ---")

@app.on_event("startup")
def startup_event():
    """
    Loads the RAG pipeline. Checks for a pre-processed text file first.
    If not found, it runs the OCR process to create it.
    """
    global rag_chain, memory, STARTUP_ERROR_MESSAGE
    
    try:
        load_dotenv()
        google_api_key = os.getenv("GOOGLE_API_KEY")

        if not google_api_key:
            STARTUP_ERROR_MESSAGE = "GOOGLE_API_KEY not found in .env file."
            print(f"ERROR: {STARTUP_ERROR_MESSAGE}")
            return
        
        if TESSERACT_CMD_PATH and os.path.exists(TESSERACT_CMD_PATH):
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD_PATH
        else:
            print("--- INFO: TESSERACT_CMD_PATH not set, assuming Tesseract is in system PATH. ---")

        # ... (Omitted unchanged file checking logic for brevity)
        if os.path.exists(OCR_TEXT_PATH):
             docs = load_text_from_file(OCR_TEXT_PATH)
        elif os.path.exists(PDF_PATH):
             ocr_pdf_with_tesseract(PDF_PATH, OCR_TEXT_PATH)
             docs = load_text_from_file(OCR_TEXT_PATH)
        else:
             STARTUP_ERROR_MESSAGE = f"CRITICAL ERROR: Neither '{OCR_TEXT_PATH}' nor '{PDF_PATH}' were found."
             print(f"ERROR: {STARTUP_ERROR_MESSAGE}")
             return

        print("--- Step 2: Setting up Parent Document Retriever... ---")
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
        
        vectorstore = FAISS.from_documents(docs, embeddings)
        store = InMemoryStore()

        parent_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )
        
        print("--- Step 3: Adding documents to the retriever... ---")
        parent_retriever.add_documents(docs, ids=None) 
        print("--- Documents added and indexed. ---")
        print("--- Step 4: Setting up LLM and prompt... ---")
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=google_api_key, temperature=0.3)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        SYSTEM_TEMPLATE = """
        You are a highly skilled assistant for answering questions from a Bengali textbook.
        Your primary task is to synthesize information from the provided context to form a complete and accurate answer.
        Carefully connect different pieces of information from the context, even if they are separated.
        If a full answer can be constructed from the context, provide it concisely.
        If the context does not contain the necessary information to answer the question, and only then, state that the information is not available in the text.
        Answer in the same language as the user's question. answer in short if possible one or two word.

        Context: {context}
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        # --- 3. WRAP THE BASE RETRIEVER WITH MULTI-QUERY ---
        print("--- Step 5: Wrapping with MultiQueryRetriever for enhanced searching... ---")
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=parent_retriever,
            llm=llm
        )
        
        document_chain = create_stuff_documents_chain(llm, prompt)

        rag_chain = (
            RunnablePassthrough.assign(context=(lambda x: x["question"]) | multi_query_retriever)
            .assign(answer=document_chain)
        )
        
        print("--- 🚀 RAG Pipeline with Multi-Query is Ready! ---")

    except Exception as e:
        error_details = f"Error details: {e}"
        print(f"!!! AN ERROR OCCURRED DURING RAG PIPELINE SETUP !!!\n{error_details}")
        STARTUP_ERROR_MESSAGE = str(e)

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Serves the main chat page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_class=HTMLResponse)
async def post_chat(request: Request, user_input: str = Form(...)):
    """Handles the chat logic, processes user input, and returns the bot's response."""
    if not rag_chain:
        error_msg = STARTUP_ERROR_MESSAGE or "Chatbot is not initialized. Check server logs."
        error_html = f"<div class='bot-message'><b>Chatbot failed to initialize.</b><br>Please check the server logs.<br><br><b>Error:</b> {error_msg}</div>"
        return HTMLResponse(error_html)

    chat_history = memory.buffer_as_messages
    response = await rag_chain.ainvoke({
        "question": user_input,
        "chat_history": chat_history
    })
    bot_answer = response['answer']

    memory.save_context({"question": user_input}, {"answer": bot_answer})
    return HTMLResponse(f"<div class='bot-message'>{bot_answer}</div>")


