from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import ConversationalRetrievalChain
import os
import warnings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from typing import List, Dict, Any

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM and Embeddings
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    groq_api_key=groq_api_key,
)

# Using HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Enhanced prompt template
prompt_template = """
You are a helpful assistant. Answer the user's question using the given context from the documents.
Only use information from the context. If the answer is not found in the context, say: "Information not available in the provided documents."

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above:
"""

# Vectorstore directory
VECTOR_DIR = "chroma_index"
os.makedirs(VECTOR_DIR, exist_ok=True)

# Store for chat sessions
store = {}

# Statistics storage
document_stats = {
    "total_documents": 0,
    "total_chunks": 0,
    "total_pages": 0,
    "processing_time": 0,
    "files_processed": []
}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Get or create chat session history"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def langchain_document_loader(directory_path: str):
    """Load documents from directory and count them"""
    documents = []
    total_pages = 0
    
    print(f"📁 Loading documents from: {directory_path}")
    
    if not os.path.exists(directory_path):
        print(f"❌ Directory not found: {directory_path}")
        return documents
    
    pdf_files = [f for f in os.listdir(directory_path) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        filepath = os.path.join(directory_path, pdf_file)
        try:
            loader = PyMuPDFLoader(filepath)
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata["source"] = pdf_file
                doc.metadata["file_path"] = filepath
            
            documents.extend(docs)
            total_pages += len(docs)
            document_stats["files_processed"].append(pdf_file)
            
            print(f"✅ Loaded {pdf_file}: {len(docs)} pages")
            
        except Exception as e:
            print(f"❌ Error loading {pdf_file}: {str(e)}")
    
    document_stats["total_documents"] = len(pdf_files)
    document_stats["total_pages"] = total_pages
    
    print(f"\n📊 Total documents loaded: {len(documents)}")
    print(f"📊 Total pages: {total_pages}")
    
    return documents

def create_chunks_with_stats(documents: List, chunk_size: int = 1600, chunk_overlap: int = 200):
    """Create chunks and provide statistics"""
    print(f"\n🔄 Creating text chunks...")
    start_time = time.time()
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Split documents
    chunks = text_splitter.split_documents(documents=documents)
    
    processing_time = time.time() - start_time
    document_stats["total_chunks"] = len(chunks)
    document_stats["processing_time"] = processing_time
    
    print(f"📊 Number of chunks created: {len(chunks)}")
    print(f"⏱️ Processing time: {processing_time:.2f} seconds")
    
    return chunks

def cosine_similarity_search(query: str, chunks: List, embeddings_model, top_k: int = 5):
    """Perform similarity search using cosine similarity with numpy"""
    print(f"\n🔍 Performing cosine similarity search...")
    
    # Get query embedding
    query_embedding = embeddings_model.embed_query(query)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    # Get chunk embeddings
    chunk_texts = [chunk.page_content for chunk in chunks]
    chunk_embeddings = embeddings_model.embed_documents(chunk_texts)
    chunk_embeddings = np.array(chunk_embeddings)
    
    # Calculate cosine similarities
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    
    # Get top-k most similar chunks
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "chunk": chunks[idx],
            "similarity": similarities[idx],
            "index": idx
        })
    
    print(f"🎯 Found {len(results)} relevant chunks")
    for i, result in enumerate(results):
        print(f"   {i+1}. Similarity: {result['similarity']:.4f}")
    
    return results

def mmr_search(query: str, chunks: List, embeddings_model, k: int = 10, lambda_mult: float = 0.5):
    """Maximal Marginal Relevance (MMR) search for diverse results"""
    print(f"\n🎯 Performing MMR search (k={k}, lambda={lambda_mult})...")
    
    # Get query embedding
    query_embedding = embeddings_model.embed_query(query)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    # Get all chunk embeddings
    chunk_texts = [chunk.page_content for chunk in chunks]
    chunk_embeddings = embeddings_model.embed_documents(chunk_texts)
    chunk_embeddings = np.array(chunk_embeddings)
    
    # Calculate similarities to query
    query_similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    
    # Initialize
    selected_indices = []
    remaining_indices = list(range(len(chunks)))
    
    # Select first document (most similar to query)
    first_idx = np.argmax(query_similarities)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Select remaining documents using MMR
    for _ in range(min(k - 1, len(remaining_indices))):
        mmr_scores = []
        
        for idx in remaining_indices:
            # Relevance to query
            relevance = query_similarities[idx]
            
            # Maximum similarity to already selected documents
            if selected_indices:
                selected_embeddings = chunk_embeddings[selected_indices]
                current_embedding = chunk_embeddings[idx].reshape(1, -1)
                max_similarity = np.max(cosine_similarity(current_embedding, selected_embeddings))
            else:
                max_similarity = 0
            
            # MMR score
            mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_similarity
            mmr_scores.append(mmr_score)
        
        # Select document with highest MMR score
        best_idx = remaining_indices[np.argmax(mmr_scores)]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    # Prepare results
    results = []
    for idx in selected_indices:
        results.append({
            "chunk": chunks[idx],
            "similarity": query_similarities[idx],
            "index": idx
        })
    
    print(f"🎯 MMR selected {len(results)} diverse chunks")
    for i, result in enumerate(results):
        print(f"   {i+1}. Similarity: {result['similarity']:.4f}")
    
    return results

def get_vectorstore_for_file(filename: str):
    """Get vectorstore for specific file"""
    file_vector_path = os.path.join(VECTOR_DIR, filename.replace('.pdf', ''))
    os.makedirs(file_vector_path, exist_ok=True)
    return Chroma(persist_directory=file_vector_path, embedding_function=embeddings)

def create_vectorstore(docs, filename: str):
    """Create and persist vectorstore"""
    print(f"🗄️ Creating vectorstore for {filename}...")
    vectordb = get_vectorstore_for_file(filename)
    vectordb.add_documents(docs)
    vectordb.persist()
    print(f"✅ Vectorstore created and persisted for {filename}")

def load_and_process_documents(directory_path: str):
    """Complete document loading and processing pipeline"""
    print("🚀 Starting document processing pipeline...")
    
    # Load documents
    documents = langchain_document_loader(directory_path)
    
    if not documents:
        print("❌ No documents found to process")
        return None, None
    
    # Create chunks with statistics
    chunks = create_chunks_with_stats(documents)
    
    # Group chunks by source file for vectorstore creation
    chunks_by_file = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        if source not in chunks_by_file:
            chunks_by_file[source] = []
        chunks_by_file[source].append(chunk)
    
    # Create vectorstores for each file
    for filename, file_chunks in chunks_by_file.items():
        create_vectorstore(file_chunks, filename)
    
    return documents, chunks

def get_enhanced_conversational_chain():
    """Get enhanced conversational chain with better retrieval"""
    all_sources = os.listdir(VECTOR_DIR)
    
    if not all_sources:
        return None
    
    # Use the first available source for simplicity
    # In a production system, you might want to merge multiple sources
    source = all_sources[0]
    db = get_vectorstore_for_file(source)
    
    # Enhanced retriever with MMR
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "lambda_mult": 0.5,
            "fetch_k": 20
        }
    )
    
    # Contextual compression for better quality
    compressor = LLMChainExtractor.from_llm(llm)
    compressed_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compressed_retriever,
        return_source_documents=True,
        verbose=False,
    )

def get_answer(query: str, use_mmr: bool = True):
    """Get answer using the enhanced RAG system"""
    print(f"\n❓ Processing query: {query}")
    
    chain = get_enhanced_conversational_chain()
    
    if not chain:
        return "❌ No documents available to answer the query."
    
    try:
        # Use default session for simplicity
        result = chain.invoke({"question": query, "chat_history": []})
        
        # Extract sources
        sources = set()
        for doc in result["source_documents"]:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            sources.add(f"{source} (page {page})")
        
        answer = result['answer'].strip()
        source_info = f"📄 Source(s): {', '.join(sources)}"
        
        return f"{answer}\n\n{source_info}"
        
    except Exception as e:
        return f"❌ Error processing query: {str(e)}"

def get_stats():
    """Get processing statistics"""
    return document_stats

def direct_similarity_search(query: str, directory_path: str = None):
    """Direct similarity search without using vectorstore"""
    if directory_path:
        documents, chunks = load_and_process_documents(directory_path)
        if not chunks:
            return "❌ No documents found for similarity search"
    else:
        print("❌ Please provide directory path for direct similarity search")
        return "❌ Directory path required"
    
    # Perform cosine similarity search
    results = cosine_similarity_search(query, chunks, embeddings, top_k=5)
    
    # Format results
    response = f"🔍 Similarity Search Results for: '{query}'\n\n"
    for i, result in enumerate(results, 1):
        chunk = result["chunk"]
        similarity = result["similarity"]
        source = chunk.metadata.get("source", "Unknown")
        page = chunk.metadata.get("page", "Unknown")
        
        response += f"{i}. **{source}** (Page {page}) - Similarity: {similarity:.4f}\n"
        response += f"   {chunk.page_content[:200]}...\n\n"
    
    return response

def direct_mmr_search(query: str, directory_path: str = None):
    """Direct MMR search without using vectorstore"""
    if directory_path:
        documents, chunks = load_and_process_documents(directory_path)
        if not chunks:
            return "❌ No documents found for MMR search"
    else:
        print("❌ Please provide directory path for direct MMR search")
        return "❌ Directory path required"
    
    # Perform MMR search
    results = mmr_search(query, chunks, embeddings, k=5)
    
    # Format results
    response = f"🎯 MMR Search Results for: '{query}'\n\n"
    for i, result in enumerate(results, 1):
        chunk = result["chunk"]
        similarity = result["similarity"]
        source = chunk.metadata.get("source", "Unknown")
        page = chunk.metadata.get("page", "Unknown")
        
        response += f"{i}. **{source}** (Page {page}) - Similarity: {similarity:.4f}\n"
        response += f"   {chunk.page_content[:200]}...\n\n"
    
    return response

# Example usage
if __name__ == "__main__":
    # Test the enhanced RAG system
    TMP_DIR = "./documents"  # Change this to your documents directory
    
    # Load and process documents
    documents, chunks = load_and_process_documents(TMP_DIR)
    
    if documents and chunks:
        print("\n" + "="*50)
        print("📊 PROCESSING STATISTICS")
        print("="*50)
        stats = get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Test query
        test_query = "What is the main topic of the document?"
        print(f"\n🔍 Testing query: {test_query}")
        
        # Test regular RAG
        answer = get_answer(test_query)
        print(f"\n📝 RAG Answer:\n{answer}")
        
        # Test direct similarity search
        print(f"\n🔍 Testing direct similarity search...")
        sim_results = direct_similarity_search(test_query, TMP_DIR)
        print(sim_results)
        
        # Test MMR search
        print(f"\n🎯 Testing MMR search...")
        mmr_results = direct_mmr_search(test_query, TMP_DIR)
        print(mmr_results)