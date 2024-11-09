from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from app.database import SessionLocal, engine
from app.models import Base, PDFDocument
from app.document_processing import process_pdf_text, get_answer_from_pdf  # Updated import based on file name
import shutil
import os

app = FastAPI()

@app.get('/')
async def root():
    return RedirectResponse(url = '/docs')

# Create all the database tables
Base.metadata.create_all(bind=engine)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for getting the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# PDF Upload Endpoint
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        # Ensure the uploaded_pdfs directory exists
        os.makedirs('uploaded_pdfs', exist_ok=True)

        # Save the uploaded PDF file locally
        file_location = f"uploaded_pdfs/{file.filename}"
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)

        # Extract text from the PDF
        pdf_text = process_pdf_text(file_location)

        # Store PDF metadata and content in the database
        new_pdf = PDFDocument(name=file.filename, content=pdf_text)
        db.add(new_pdf)
        db.commit()
        db.refresh(new_pdf)

        return {"pdf_id": new_pdf.id, "filename": file.filename, "message": "PDF uploaded successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading PDF: {e}")

# Question Answering Endpoint
@app.post("/ask_question/")
async def ask_question(pdf_id: int = Form(...), question: str = Form(...), db: Session = Depends(get_db)):
    # Retrieve PDF from the database by ID
    pdf = db.query(PDFDocument).filter(PDFDocument.id == pdf_id).first()
    if not pdf:
        raise HTTPException(status_code=404, detail="PDF not found")

    # Get an answer using LangChain based on the PDF content
    answer = get_answer_from_pdf(pdf.content, question)

    return {"pdf_id": pdf_id, "answer": answer}
