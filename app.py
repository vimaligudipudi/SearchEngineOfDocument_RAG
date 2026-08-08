# ========== IMPORTS ==========
print("IMPORTING MODULES")
print("=" * 50)

import os
import sys
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Flask imports
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# LlamaIndex core imports
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Document,
    Settings
)
from llama_index.core.node_parser import SimpleNodeParser

# Groq LLM
from llama_index.llms.groq import Groq

# Embeddings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("All modules imported successfully!")

# ========== EXACT RAG SYSTEM CLASS (DEFINE THIS FIRST) ==========
class ExactAnswerRAG:
    def __init__(self, index, llm):
        self.index = index
        self.llm = llm
        self.document_understanding = None

    def understand_document(self):
        """Thoroughly understand the document first"""
        print("Understanding document completely...")

        # Get comprehensive understanding
        all_nodes = list(self.index.docstore.docs.values())

        if not all_nodes:
            print("No documents loaded to understand!")
            return False

        # Extract all key information (limit for efficiency)
        sample_text = ''.join([node.text[:500] for node in all_nodes[:5]])

        understanding_prompt = f"""
        I need to completely understand this document. Here are sample content chunks:

        {sample_text}

        Based on this content, provide a comprehensive understanding that covers:
        1. Main topics and subjects
        2. Key concepts and definitions
        3. Overall document structure

        COMPREHENSIVE UNDERSTANDING:
        """

        try:
            understanding = self.llm.complete(understanding_prompt)
            self.document_understanding = str(understanding)
            print("Document fully understood!")
            return True
        except Exception as e:
            print(f"Understanding failed: {e}")
            return False

    def ask_exact_question(self, question):
        """Ask questions and get exact answers based on full understanding"""
        if not self.document_understanding:
            return "Please understand the document first using understand_document()"

        print(f"Exact Question: {question}")

        # Get relevant context
        retriever = self.index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(question)

        # Use exact context
        exact_context = "\n\n".join([node.text for node in nodes])

        exact_prompt = f"""
        DOCUMENT CONTEXT:
        {exact_context}

        QUESTION: {question}

        REQUIREMENTS:
        1. Answer MUST be based on the document content above
        2. Be factual and accurate
        3. If information is not in the document, say "This information is not found in the document"

        ANSWER:
        """

        response = self.llm.complete(exact_prompt)
        return str(response)

# ========== FLASK APP SETUP ==========
app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ========== GROQ API SETUP ==========
print("\n GROQ API SETUP")
print("=" * 50)

# Your Groq API Key
GROQ_API_KEY = "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"

# Set environment variable
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
print(" Groq API key set successfully!")

# ========== MODEL INITIALIZATION ==========
print("\n INITIALIZING GROQ LLM AND EMBEDDINGS")
print("=" * 50)

class RAGSystem:
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.index = None
        self.query_engine = None
        self.exact_rag = None
        self.nodes = []
        self.knowledge_base = {
            "concepts": [],
            "examples": [],
            "definitions": [],
            "procedures": [],
            "syntax": []
        }
        self.document_name = None
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize Groq LLM and embedding models"""
        try:
            # Initialize Groq LLM
            self.llm = Groq(
                model="llama-3.1-8b-instant",
                api_key=GROQ_API_KEY,
                temperature=0.1,
                max_tokens=2048,
                context_window=32768
            )
            
            # Initialize Embeddings
            self.embeddings = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Configure global settings
            Settings.llm = self.llm
            Settings.embed_model = self.embeddings
            Settings.chunk_size = 512
            Settings.chunk_overlap = 50
            
            # Create initial empty index
            self.index = VectorStoreIndex([], embed_model=self.embeddings)
            self.query_engine = self.index.as_query_engine(llm=self.llm)
            
            # Initialize Exact RAG
            self.exact_rag = ExactAnswerRAG(self.index, self.llm)
            
            print(" Models initialized successfully!")
            return True
        except Exception as e:
            print(f" Model initialization failed: {e}")
            return False

# Initialize the RAG system
rag_system = RAGSystem()

# ========== HELPER FUNCTIONS ==========

def process_document(file_path):
    """Process a document and add it to the RAG system"""
    try:
        print(f" Processing document: {file_path}")
        
        # For TXT files
        if file_path.lower().endswith('.txt'):
            print(" Processing text file...")
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            documents = [Document(text=text_content)]

        # For Word documents
        elif file_path.lower().endswith(('.docx', '.doc')):
            print(" Processing Word document...")
            import docx2txt
            text_content = docx2txt.process(file_path)
            documents = [Document(text=text_content)]

        # For PDF files
        else:
            print(" Processing PDF document...")
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

        print(f" Loaded {len(documents)} document sections")

        # Chunk the document
        parser = SimpleNodeParser.from_defaults(
            chunk_size=512,
            chunk_overlap=50
        )
        new_nodes = parser.get_nodes_from_documents(documents)
        print(f" Created {len(new_nodes)} chunks from document")

        # Add to index
        for node in new_nodes:
            rag_system.index.insert_nodes([node])

        # Update query engine
        rag_system.query_engine = rag_system.index.as_query_engine(llm=rag_system.llm)

        # Rebuild knowledge base
        rag_system.knowledge_base = create_knowledge_base()

        # Understand document
        rag_system.exact_rag.understand_document()

        return {
            "success": True,
            "chunks_created": len(new_nodes),
            "total_chunks": len(rag_system.index.docstore.docs),
            "document_name": os.path.basename(file_path)
        }

    except Exception as e:
        print(f" Error processing document: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def create_knowledge_base():
    """Create a structured knowledge base from the document"""
    all_nodes = list(rag_system.index.docstore.docs.values())

    knowledge_structure = {
        "concepts": [],
        "examples": [],
        "definitions": [],
        "procedures": [],
        "syntax": []
    }

    for node in all_nodes:
        text = node.text

        # Simple classification
        if any(keyword in text.lower() for keyword in ['def ', 'function', 'method']):
            knowledge_structure["syntax"].append(text)
        elif any(keyword in text.lower() for keyword in ['example', 'output', 'result']):
            knowledge_structure["examples"].append(text)
        elif any(keyword in text.lower() for keyword in ['definition', 'means', 'is a']):
            knowledge_structure["definitions"].append(text)
        elif any(keyword in text.lower() for keyword in ['step', 'process', 'procedure']):
            knowledge_structure["procedures"].append(text)
        else:
            knowledge_structure["concepts"].append(text)

    print(" Knowledge base created!")
    for category, items in knowledge_structure.items():
        print(f"   • {category}: {len(items)} items")
    
    return knowledge_structure

# ========== API ROUTES ==========

@app.route('/')
def home():
    """Home route - API status"""
    return jsonify({
        "status": "success",
        "message": "Groq RAG Backend Server is running!",
        "endpoints": {
            "GET /": "API status",
            "POST /upload": "Upload and process document",
            "POST /ask": "Ask a question",
            "POST /ask-exact": "Ask exact question",
            "GET /reset": "Reset system",
            "GET /status": "System status"
        }
    })

@app.route('/status', methods=['GET'])
def get_status():
    """Get system status"""
    return jsonify({
        "status": "success",
        "system_status": {
            "llm_initialized": rag_system.llm is not None,
            "embeddings_initialized": rag_system.embeddings is not None,
            "index_created": rag_system.index is not None,
            "documents_loaded": len(rag_system.index.docstore.docs) if rag_system.index else 0,
            "document_name": rag_system.document_name,
            "knowledge_base_items": sum(len(v) for v in rag_system.knowledge_base.values()),
            "document_understood": rag_system.exact_rag.document_understanding is not None if rag_system.exact_rag else False
        }
    })

@app.route('/upload', methods=['POST'])
def upload_document():
    """Upload and process a document"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file provided"}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({
                "status": "error", 
                "message": "Invalid file type. Allowed: pdf, docx, doc, txt"
            }), 400
        
        # Secure filename and save
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Process the document
        result = process_document(file_path)
        
        if result['success']:
            rag_system.document_name = filename
            return jsonify({
                "status": "success",
                "message": "Document processed successfully",
                "data": result
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Document processing failed",
                "error": result['error']
            }), 500
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Upload failed",
            "error": str(e)
        }), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Ask a question to the RAG system"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({"status": "error", "message": "Question is required"}), 400
        
        question = data['question']
        
        if not rag_system.index or len(rag_system.index.docstore.docs) == 0:
            return jsonify({
                "status": "error", 
                "message": "No documents loaded. Please upload a document first."
            }), 400
        
        # Ask question
        response = rag_system.query_engine.query(question)
        
        return jsonify({
            "status": "success",
            "question": question,
            "answer": str(response),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Question processing failed",
            "error": str(e)
        }), 500

@app.route('/ask-exact', methods=['POST'])
def ask_exact_question():
    """Ask an exact question using the enhanced RAG system"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({"status": "error", "message": "Question is required"}), 400
        
        question = data['question']
        
        if not rag_system.index or len(rag_system.index.docstore.docs) == 0:
            return jsonify({
                "status": "error", 
                "message": "No documents loaded. Please upload a document first."
            }), 400
        
        # Ask exact question
        response = rag_system.exact_rag.ask_exact_question(question)
        
        return jsonify({
            "status": "success",
            "question": question,
            "answer": str(response),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Exact question processing failed",
            "error": str(e)
        }), 500

@app.route('/reset', methods=['GET'])
def reset_system():
    """Reset the entire RAG system"""
    try:
        # Reinitialize the system
        rag_system.__init__()
        
        # Clear uploads directory
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        
        return jsonify({
            "status": "success",
            "message": "System reset successfully"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "System reset failed",
            "error": str(e)
        }), 500

# ========== START SERVER ==========
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print(" GROQ RAG BACKEND SERVER STARTING...")
    print("=" * 60)
    print(" API Endpoints:")
    print("   GET  /          - API status")
    print("   GET  /status    - System status") 
    print("   POST /upload    - Upload document")
    print("   POST /ask       - Ask question")
    print("   POST /ask-exact - Ask exact question")
    print("   GET  /reset     - Reset system")
    print("=" * 60)
    print(" Server running on: http://localhost:5000")
    print("=" * 60)
    

    app.run(host='0.0.0.0', port=5000, debug=True)
