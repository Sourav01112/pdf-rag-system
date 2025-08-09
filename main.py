import ollama
import time
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough

print("Welcome to the Optimized PDF RAG System! >>")


def time_operation(operation_name, func):
    start_time = time.time()
    result = func()
    end_time = time.time()
    print(f"{operation_name}: {end_time - start_time:.2f} seconds")
    return result


doc_pwd = "./data/America_Data_Future_Final.pdf"
model = "llama3.2"
remote_ollama = "http://192.168.1.11:11434"

print(f"Loading PDF from: {doc_pwd}")


def load_pdf():
    if doc_pwd:
        loader = PyMuPDFLoader(file_path=doc_pwd)
        return loader.load()
    return []


data = time_operation("PDF Loading", load_pdf)


def split_text():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    print(f":chunks created:::::: {len(chunks)}")
    return chunks


chunks = time_operation("Text Splitting", split_text)


def create_embeddings():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=remote_ollama,
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="optimized-rag",
        persist_directory="./chroma_db",
    )

    print(f"Stored docs:::::: {vector_store._collection.count()}")
    return vector_store


vector_store = time_operation("Embedding Creation", create_embeddings)


def setup_rag():
    llm = ChatOllama(
        model=model,
        base_url=remote_ollama,
        num_ctx=2048,
        temperature=0.2,
        top_p=0.8,
        repeat_penalty=1.1,
        num_gpu=1,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    template = """Use the following context to answer the question concisely.

Context: {context}

Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_pipeline = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_pipeline


rag_pipeline = time_operation("RAG Setup", setup_rag)


def process_query(question):
    print(f"\nProcessing query:::::::: {question}")
    return rag_pipeline.invoke(question)


print("\n" + "=" * 50)
print("Pipeline ready! Processing queries...")

queries = [
    "Does the document mention any specific technologies?",
    "what is the document about?",
    "What is the future of America in terms of data and technology?",
]

for query in queries:
    response = time_operation(f"Query: '{query}'", lambda: process_query(query))
    print(f"Response::::::::: {response}\n")
    print("-" * 50)


# ## 1. Ingest PDF Files
# # 2. Extract Text from PDF Files and split into small chunks
# # 3. Send the chunks to the embedding model
# # 4. Save the embeddings to a vector database
# # 5. Perform similarity search on the vector database to find similar documents
# # 6. retrieve the similar documents and present them to the user
# ## run pip install -r requirements.txt to install the required packages

# import ollama
# from langchain_community.document_loaders import UnstructuredPDFLoader
# from langchain_ollama import OllamaEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_ollama import ChatOllama
# from langchain_core.runnables import RunnablePassthrough
# from langchain.retrievers.multi_query import MultiQueryRetriever


# print("Welcome to the PDF Ingestion and Embedding System!")

# doc_pwd = "./data/America_Data_Future_Final.pdf"
# model = "llama3.2"

# print(f"Loading PDF from: {doc_pwd}")
# if doc_pwd:
#     loader = UnstructuredPDFLoader(file_path=doc_pwd)
#     data = loader.load()
# else:
#     print("No document path provided. Please provide a valid PDF file path.")
#     data = []


# content = data[0].page_content
# # print(content[:100])
# # Ingest PDF Files ENDS =======


# # ========= Extract Text from PDF Files and split into small chunks


# text_splitter_func = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
# chunks = text_splitter_func.split_documents(data)
# print(f"Number of chunks created: {len(chunks)}")
# print(":split done:::::chunks created::::")
# # Extract Text from PDF Files and split into small chunks ENDS =======


# #  ======== Send the chunks to the embedding model

# remote_ollama = "http://192.168.1.11:11434"

# # ollama.pull("nomic-embed-text")

# embeddings = OllamaEmbeddings(
#     model="nomic-embed-text",
#     base_url=remote_ollama,  # use this
# )

# vector_store = Chroma.from_documents(
#     documents=chunks,
#     embedding=embeddings,
#     collection_name="simple-rag",
# )

# print(":Embeddings stored in vector store::::::", vector_store)
# print(":Stored docs::::::", vector_store._collection.count())
# # results = vector_store.similarity_search("test query", k=2)
# # for doc in results:
# #     print("---")
# #     print(doc.page_content[:200])
# # Send the chunks to the embedding model ENDS =======


# # Retrieval and Querying
# llm = ChatOllama(model=model, base_url=remote_ollama)

# base_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# QUERY_PROMPT = PromptTemplate(
#     input_variables=["question"],
#     template="""You are an AI language model assistant. Your task is to generate five
#     different versions of the given user question to retrieve relevant documents from
#     a vector database. By generating multiple perspectives on the user question, your
#     goal is to help the user overcome some of the limitations of the distance-based
#     similarity search. Provide these alternative questions separated by newlines.
#     Original question: {question}""",
# )

# retriever = MultiQueryRetriever.from_llm(
#     llm=llm,
#     retriever=base_retriever,
#     prompt=QUERY_PROMPT,
# )


# # RAG prompting
# template = (
#     "You are a helpful assistant. Use the context to answer the question.\n"
#     "{context}\n"
#     "Question: {question}\n"
#     "Answer:"
# )
# prompt = ChatPromptTemplate.from_template(template)

# rag_pipeline = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )
# print("Starting the pipeline.....")
# resp = rag_pipeline.invoke("what is the document about?")
# # resp = rag_pipeline.invoke("What is the future of America in terms of data and technology?")
# print("Response from RAG pipeline:")
# print(resp)
