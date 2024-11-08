from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.database import Base

class PDFDocument(Base):
    __tablename__ = "pdf_documents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    content = Column(String)  # Storing extracted PDF text
    created_at = Column(DateTime(timezone=True), server_default=func.now())  # Automatically set the creation time
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())