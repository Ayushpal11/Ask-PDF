# from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import RedirectResponse
# from sqlalchemy.orm import Session
# from app.database import SessionLocal, engine
# from app.models import Base, PDFDocument
# from app.document_processing import process_pdf_text, get_answer_from_pdf  # Updated import based on file name
# import shutil
# import os

# app = FastAPI()

# @app.get('/')
# async def root():
#     return RedirectResponse(url = '/docs')

# # Create all the database tables
# Base.metadata.create_all(bind=engine)

# # CORS Middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Dependency for getting the database session
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# # PDF Upload Endpoint
# @app.post("/upload_pdf/")
# async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
#     try:
#         # Ensure the uploaded_pdfs directory exists
#         os.makedirs('uploaded_pdfs', exist_ok=True)

#         # Save the uploaded PDF file locally
#         file_location = f"uploaded_pdfs/{file.filename}"
#         with open(file_location, "wb+") as file_object:
#             shutil.copyfileobj(file.file, file_object)

#         # Extract text from the PDF
#         pdf_text = process_pdf_text(file_location)

#         # Store PDF metadata and content in the database
#         new_pdf = PDFDocument(name=file.filename, content=pdf_text)
#         db.add(new_pdf)
#         db.commit()
#         db.refresh(new_pdf)

#         return {"pdf_id": new_pdf.id, "filename": file.filename, "message": "PDF uploaded successfully!"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error uploading PDF: {e}")

# # Question Answering Endpoint
# @app.post("/ask_question/")
# async def ask_question(pdf_id: int = Form(...), question: str = Form(...), db: Session = Depends(get_db)):
#     # Retrieve PDF from the database by ID
#     pdf = db.query(PDFDocument).filter(PDFDocument.id == pdf_id).first()
#     if not pdf:
#         raise HTTPException(status_code=404, detail="PDF not found")

#     # Get an answer using LangChain based on the PDF content
#     answer = get_answer_from_pdf(pdf.content, question)

#     return {"pdf_id": pdf_id, "answer": answer}

# # Delete PDF Endpoint:: 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


import os
import io
import shutil
from datetime import datetime
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from databases import Database
from pdfminer.high_level import extract_text
from langchain.chains.question_answering import load_qa_chain
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Initialize FastAPI app
app = FastAPI()

# Set up CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Configuration
DATABASE_URL = "sqlite:///./test.db"  
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
metadata = MetaData()

# Define the PDF document model for the database
class PDFDocument(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    upload_date = Column(String)
    content = Column(String)

# Create tables in the database
Base.metadata.create_all(bind=engine)

# Utility functions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def save_uploaded_file(file: UploadFile, destination: str):
    with open(destination, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

def process_pdf_text(pdf_path: str) -> str:
    with open(pdf_path, "rb") as pdf_file:
        text = extract_text(pdf_file)
    return text

# def get_answer_from_pdf(pdf_content: str, question: str) -> str:
#     # Set Hugging Face API token
#     os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

#     # Prepare prompt for the model
#     prompt = f"###Instruction\n{question}\n###Response\n"

#     # Initialize tokenizer and model, ensuring consistent device assignment
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
#     model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)

#     # Tokenize input prompt, removing token_type_ids if not supported
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     if 'token_type_ids' in inputs:
#         del inputs['token_type_ids']

#     # Generate response tokens with specified parameters
#     tokens = model.generate(**inputs, max_new_tokens=200, temperature=0.8)

#     # Decode the generated tokens to get the answer text
#     answer = tokenizer.decode(tokens[0], skip_special_tokens=True)

#     return answer

def get_answer_from_pdf(pdf_content: str, question: str) -> str:
    # Set Hugging Face API token
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

    # Prepare a well-formatted prompt for the model
    prompt = f"### Instruction: Answer the following question based on the PDF content provided.\nContent:\n{pdf_content[:1000]}...\nQuestion: {question}\n### Response:"

    # Initialize tokenizer and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)

    # Tokenize input prompt, ensuring it’s loaded onto the correct device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    if 'token_type_ids' in inputs:
        del inputs['token_type_ids']  # Remove if the model doesn’t support token_type_ids

    # Generate response tokens with refined parameters
    tokens = model.generate(
        **inputs, 
        max_new_tokens=250,     # Increased token limit for more detailed responses
        temperature=0.7,        # Lowered temperature for more focused responses
        num_beams=3,            # Adding beam search for better response quality
        early_stopping=True     # Stop generation when a coherent response is reached
    )

    # Decode the generated tokens to get the answer text
    answer = tokenizer.decode(tokens[0], skip_special_tokens=True)

    # Log the generated answer for debugging
    print(f"Generated Answer: {answer}")

    return answer if answer.strip() else "No relevant information found in the document."

# Define Pydantic models for requests and responses
class PDFUploadResponse(BaseModel):
    pdf_id: int
    filename: str
    message: str

class QuestionInput(BaseModel):
    pdf_id: int
    question: str

class AnswerResponse(BaseModel):
    pdf_id: int
    answer: str

# PDF Upload Endpoint
@app.post("/upload_pdf/", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...), db: SessionLocal = Depends(get_db)):
    try:
        os.makedirs('uploaded_pdfs', exist_ok=True)
        file_location = f"uploaded_pdfs/{file.filename}"
        
        await save_uploaded_file(file, file_location)

        pdf_text = process_pdf_text(file_location)
        
        new_pdf = PDFDocument(filename=file.filename, upload_date=str(datetime.now()), content=pdf_text)
        db.add(new_pdf)
        db.commit()
        db.refresh(new_pdf)

        return {"pdf_id": new_pdf.id, "filename": file.filename, "message": "PDF uploaded successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {e}")

# Question Answering Endpoint
@app.post("/ask_question/", response_model=AnswerResponse)
async def ask_question(data: QuestionInput, db: SessionLocal = Depends(get_db)):
    pdf = db.query(PDFDocument).filter(PDFDocument.id == data.pdf_id).first()
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF not found")

    answer = get_answer_from_pdf(pdf.content, data.question)

    return {"pdf_id": data.pdf_id, "answer": answer}

@app.get('/')
async def root():
    return RedirectResponse(url='/docs')
