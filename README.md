# PDF RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that enables intelligent question-answering over PDF documents using local LLMs and embeddings.

## 🚀 Features

- **PDF Document Ingestion**: Automatically extract and process text from PDF files
- **Intelligent Text Chunking**: Split documents into optimal chunks for processing
- **Vector Embeddings**: Generate semantic embeddings using Ollama models
- **Similarity Search**: Find relevant document sections based on query similarity
- **Question Answering**: Generate contextual answers using retrieved information
- **Local Processing**: Runs entirely on your infrastructure using Ollama


[Alt text](./image/Screenshot.png)

## 📋 Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running
- At least 8GB RAM (16GB recommended for larger documents)

## ⚡ Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd pdf-rag-pipeline

# Install required packages
pip install -r requirements.txt
```

### 2. Setup Ollama

```bash
# Pull required models
ollama pull llama3.2
ollama pull nomic-embed-text

# Start Ollama server (if not already running)
ollama serve
```

### 3. Prepare Your Document

Place your PDF file in the `./data/` directory or update the `doc_pwd` variable in the script.

### 4. Run the Pipeline

```bash
python main.py
```

## 🔧 How It Works

The RAG pipeline follows these steps:

### 1. **Ingest PDF Files**
```python
loader = UnstructuredPDFLoader(file_path=doc_pwd)
data = loader.load()
```
- Loads PDF documents using LangChain's document loaders
- Extracts text content while preserving document structure

### 2. **Extract Text and Split into Chunks**
```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(data)
```
- Splits large documents into manageable chunks
- Maintains context overlap between chunks for better retrieval

### 3. **Generate Embeddings**
```python
embeddings = OllamaEmbeddings(model="nomic-embed-text")
```
- Converts text chunks into vector embeddings
- Uses Ollama's `nomic-embed-text` model for semantic representation

### 4. **Store in Vector Database**
```python
vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)
```
- Saves embeddings in Chroma vector database
- Enables fast similarity search capabilities

### 5. **Similarity Search**
```python
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
```
- Finds most relevant document chunks for user queries
- Returns top-k similar documents based on embedding similarity

### 6. **Generate Answers**
```python
rag_pipeline = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```
- Combines retrieved context with user questions
- Uses LLM to generate comprehensive answers

## 📁 Project Structure

```
pdf-rag-pipeline/
├── main.py                 # Main RAG pipeline script
├── requirements.txt        # Python dependencies
├── data/                   # PDF documents directory
│   └── your_document.pdf
├── chroma_db/             # Vector database storage (auto-created)
└── README.md              # This file
```

## 📦 Dependencies

```txt
ollama
chromadb
pdfplumber 
langchain
langchain-core
langchain-ollama
langchain-community
langchain_text_splitters
unstructured[pdf]
fastembed
pikepdf
elevenlabs
PyMuPDFLoader
```

## ⚙️ Configuration

### Document Settings
- **PDF Path**: Update `doc_pwd` variable for your document location
- **Chunk Size**: Adjust `chunk_size` (default: 1200) for different document types
- **Chunk Overlap**: Modify `chunk_overlap` (default: 300) to maintain context

### Model Settings
- **LLM Model**: Change `model` variable (default: "llama3.2")
- **Embedding Model**: Modify embedding model in `OllamaEmbeddings`
- **Remote Ollama**: Update `remote_ollama` URL if using remote instance

### Retrieval Settings
- **Search Results**: Adjust `k` parameter in retriever (default: 4)
- **Search Type**: Choose between "similarity", "mmr", or "similarity_score_threshold"

## 🔍 Usage Examples

### Basic Query
```python
response = rag_pipeline.invoke("What is the main topic of this document?")
print(response)
```

### Multiple Queries
```python
queries = [
    "What are the key findings?",
    "What recommendations are made?",
    "Who are the main stakeholders?"
]

for query in queries:
    response = rag_pipeline.invoke(query)
    print(f"Q: {query}")
    print(f"A: {response}\n")
```

## 🚀 Performance Optimization

### Speed Improvements
- Use `PyMuPDFLoader` instead of `UnstructuredPDFLoader` for faster PDF processing
- Reduce chunk size to 800-1000 characters
- Use `llama3.2:1b` for faster inference
- Enable GPU acceleration in Ollama

### Memory Optimization
- Process documents in batches for large collections
- Use persistent vector storage to avoid re-processing
- Clear unused variables after processing

### Example Optimized Settings
```python
# Faster PDF loading
loader = PyMuPDFLoader(file_path=doc_pwd)

# Optimized chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, 
    chunk_overlap=200
)

# Persistent storage
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

## 📊 Expected Performance

| Operation | Time (Small PDF) | Time (Large PDF) |
|-----------|------------------|------------------|
| PDF Loading | 1-3 seconds | 5-15 seconds |
| Text Chunking | <1 second | 1-5 seconds |
| Embedding Creation | 10-30 seconds | 1-5 minutes |
| Query Processing | 15-30 seconds | 20-45 seconds |

## 🛠️ Troubleshooting

### Common Issues

**1. Ollama Connection Error**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama if needed
ollama serve
```

**2. Memory Issues**
- Reduce chunk size and batch size
- Use smaller models (llama3.2:1b)
- Process documents individually

**3. Slow Performance**
- Run Ollama locally instead of remote server
- Use SSD storage for vector database
- Enable GPU acceleration

**4. PDF Loading Errors**
- Install additional dependencies: `pip install unstructured[all-docs]`
- Try different PDF loaders (PyMuPDF, PDFPlumber)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM inference
- [LangChain](https://langchain.com/) for RAG pipeline framework
- [Chroma](https://www.trychroma.com/) for vector database
- [Unstructured](https://unstructured.io/) for PDF processing