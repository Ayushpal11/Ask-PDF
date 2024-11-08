import os
import fitz  # PyMuPDF for PDF text extraction
from langchain_ai21 import AI21Embeddings
from langchain_community.llms import AI21
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Access the API key from the environment
AI21_API_KEY = os.getenv("")

# PDF Text Extraction Function
def process_pdf_text(pdf_path: str) -> str:
    """Extracts and returns the text content from the PDF file."""
    try:
        with fitz.open(pdf_path) as pdf:
            text = ""
            for page in pdf:
                text += page.get_text()
        return text
    except Exception as e:
        raise Exception(f"Error processing PDF: {e}")

# Function to answer questions based on PDF content
def get_answer_from_pdf(pdf_text: str, question: str) -> str:
    """Processes the PDF content with LangChain and returns the answer to the question."""
    
    if not AI21_API_KEY:
        raise Exception("AI21 API key not found in environment variables.")

    try:
        # Initialize LangChain components with the AI21 API key
        embeddings = AI21Embeddings(ai21_api_key=AI21_API_KEY)
        llm = AI21(ai21_api_key=AI21_API_KEY, model="j2-ultra")  # or use j2-mid for a smaller model

        # Create a vector store from the embeddings
        vectorstore = Chroma.from_texts([pdf_text], embeddings)

        # Create the RetrievalQA chain using the vector store
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )

        # Get the answer
        result = chain({"query": question})
        if "result" not in result:
            raise Exception("No answer found for the given question.")
        return result["result"]
    
    except Exception as e:
        raise Exception(f"Error in question answering process: {e}")