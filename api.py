from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import shutil
import tempfile
from pathlib import Path
import json
import asyncio
from datetime import datetime

# Import your enhanced RAG system
from rag import (
    get_answer, 
    load_and_process_documents, 
    get_stats,
    direct_similarity_search,
    direct_mmr_search,
    document_stats
)

# FastAPI app initialization
app = FastAPI(
    title="Enhanced RAG Chatbot API",
    description="A powerful RAG system with document processing, similarity search, and MMR",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory
UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"
    search_type: Optional[str] = "rag"  # rag, similarity, mmr

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    processing_time: float
    search_type: str
    session_id: str

class DocumentStats(BaseModel):
    total_documents: int
    total_chunks: int
    total_pages: int
    processing_time: float
    files_processed: List[str]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

# Global variables
current_documents_dir = None
is_processing = False
processing_status = {"status": "idle", "message": "Ready to process documents"}

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0"
    )

@app.post("/upload-documents/")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload multiple PDF documents"""
    global current_documents_dir, is_processing, processing_status
    
    if is_processing:
        raise HTTPException(
            status_code=400, 
            detail="Another document processing is in progress"
        )
    
    is_processing = True
    processing_status = {"status": "uploading", "message": "Uploading documents..."}
    
    try:
        # Create temporary directory for this upload session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = UPLOAD_DIR / f"session_{timestamp}"
        session_dir.mkdir(exist_ok=True)
        
        uploaded_files = []
        
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is not a PDF"
                )
            
            # Save uploaded file
            file_path = session_dir / file.filename
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            uploaded_files.append(file.filename)
        
        # Update global documents directory
        current_documents_dir = str(session_dir)
        
        processing_status = {"status": "processing", "message": "Processing documents..."}
        
        # Process documents
        documents, chunks = load_and_process_documents(current_documents_dir)
        
        if not documents:
            processing_status = {"status": "error", "message": "No documents processed"}
            raise HTTPException(status_code=400, detail="No documents were processed")
        
        # Get statistics
        stats = get_stats()
        
        processing_status = {"status": "completed", "message": "Documents processed successfully"}
        
        return {
            "message": "Documents uploaded and processed successfully",
            "files_uploaded": uploaded_files,
            "stats": stats,
            "documents_directory": current_documents_dir
        }
        
    except Exception as e:
        processing_status = {"status": "error", "message": f"Error: {str(e)}"}
        raise HTTPException(status_code=500, detail=f"Error processing documents: {str(e)}")
    finally:
        is_processing = False

@app.get("/processing-status/")
async def get_processing_status():
    """Get current processing status"""
    return processing_status

@app.post("/query/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the processed documents"""
    global current_documents_dir
    
    if not current_documents_dir:
        raise HTTPException(
            status_code=400, 
            detail="No documents uploaded. Please upload documents first."
        )
    
    start_time = datetime.now()
    
    try:
        if request.search_type == "similarity":
            # Direct similarity search
            answer = direct_similarity_search(request.query, current_documents_dir)
            sources = []
        elif request.search_type == "mmr":
            # Direct MMR search
            answer = direct_mmr_search(request.query, current_documents_dir)
            sources = []
        else:
            # Default RAG with conversational chain
            answer = get_answer(request.query)
            
            # Extract sources from the answer
            sources = []
            if "📄 Source(s):" in answer:
                sources_text = answer.split("📄 Source(s):")[1].strip()
                answer = answer.split("📄 Source(s):")[0].strip()
                sources = [s.strip() for s in sources_text.split(",")]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            processing_time=processing_time,
            search_type=request.search_type,
            session_id=request.session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/stats/", response_model=DocumentStats)
async def get_document_stats():
    """Get document processing statistics"""
    stats = get_stats()
    return DocumentStats(**stats)

@app.post("/reset/")
async def reset_system():
    """Reset the system and clear uploaded documents"""
    global current_documents_dir, processing_status
    
    try:
        # Clean up upload directory
        if UPLOAD_DIR.exists():
            shutil.rmtree(UPLOAD_DIR)
            UPLOAD_DIR.mkdir(exist_ok=True)
        
        # Reset global variables
        current_documents_dir = None
        processing_status = {"status": "idle", "message": "System reset successfully"}
        
        # Reset document stats
        document_stats.update({
            "total_documents": 0,
            "total_chunks": 0,
            "total_pages": 0,
            "processing_time": 0,
            "files_processed": []
        })
        
        return {"message": "System reset successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting system: {str(e)}")

@app.get("/search-types/")
async def get_search_types():
    """Get available search types"""
    return {
        "search_types": [
            {
                "type": "rag",
                "name": "RAG (Retrieval Augmented Generation)",
                "description": "Full conversational RAG with contextual compression"
            },
            {
                "type": "similarity",
                "name": "Cosine Similarity Search",
                "description": "Direct similarity search using cosine similarity"
            },
            {
                "type": "mmr",
                "name": "MMR (Maximal Marginal Relevance)",
                "description": "Diverse document retrieval using MMR algorithm"
            }
        ]
    }

@app.get("/files/")
async def list_uploaded_files():
    """List currently uploaded files"""
    global current_documents_dir
    
    if not current_documents_dir or not os.path.exists(current_documents_dir):
        return {"files": [], "message": "No files uploaded"}
    
    files = []
    for file in os.listdir(current_documents_dir):
        if file.endswith('.pdf'):
            file_path = os.path.join(current_documents_dir, file)
            file_size = os.path.getsize(file_path)
            files.append({
                "name": file,
                "size": file_size,
                "size_mb": round(file_size / (1024 * 1024), 2)
            })
    
    return {"files": files, "total_files": len(files)}

@app.delete("/files/{filename}")
async def delete_file(filename: str):
    """Delete a specific uploaded file"""
    global current_documents_dir
    
    if not current_documents_dir:
        raise HTTPException(status_code=400, detail="No documents directory")
    
    file_path = os.path.join(current_documents_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        os.remove(file_path)
        return {"message": f"File {filename} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    print("🚀 Enhanced RAG Chatbot API is starting up...")
    print("📚 Ready to process documents and answer questions!")
    print("🔍 Available search types: RAG, Similarity, MMR")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    print("🛑 Enhanced RAG Chatbot API is shutting down...")

if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )