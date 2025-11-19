<<<<<<< HEAD
"""
AmbedkarGPT - Q&A System using RAG Pipeline
A command-line application that answers questions based on Dr. B.R. Ambedkar's speech
using LangChain, ChromaDB, HuggingFace Embeddings, and Ollama Mistral 7B
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Constants
SPEECH_FILE = "speech.txt"
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemma3:1b"

def load_and_split_document(file_path):
    """
    Load the text file and split it into manageable chunks
    
    Args:
        file_path (str): Path to the speech text file
    
    Returns:
        list: List of document chunks
    """
    print(f"Loading document from {file_path}...")
    
    # Load the text file
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    
    # Split the text into chunks
    # Using smaller chunk size for better retrieval on short document
    text_splitter = CharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separator="\n"
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Document split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    """
    Create embeddings and store them in ChromaDB vector store
    
    Args:
        chunks (list): List of document chunks
    
    Returns:
        Chroma: Vector store instance
    """
    print("Creating embeddings using HuggingFace model...")
    
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create and persist the vector store
    print("Storing embeddings in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    
    print(f"Vector store created and persisted to {PERSIST_DIRECTORY}")
    return vectorstore

def load_existing_vector_store():
    """
    Load an existing vector store from disk
    
    Returns:
        Chroma: Vector store instance
    """
    print("Loading existing vector store...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    
    return vectorstore

def setup_qa_chain(vectorstore):
    """
    Set up the RAG chain with Ollama LLM using LCEL (LangChain Expression Language)
    
    Args:
        vectorstore (Chroma): Vector store instance
    
    Returns:
        Runnable: Question-answering chain
    """
    print("Setting up QA chain with Ollama Mistral 7B...")
    
    # Initialize Ollama LLM
    llm = Ollama(
        model=LLM_MODEL,
        temperature=0.3  # Lower temperature for more focused answers
    )
    
    # Create a custom prompt template
    template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Keep your answer concise and directly related to the question.

Context: {context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate.from_template(template)
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create RAG chain using LCEL
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("QA chain ready!\n")
    return rag_chain, retriever

def initialize_system():
    """
    Initialize the RAG system by loading or creating the vector store
    
    Returns:
        tuple: (rag_chain, retriever)
    """
    # Check if vector store already exists
    if os.path.exists(PERSIST_DIRECTORY):
        print("Existing vector store found.")
        vectorstore = load_existing_vector_store()
    else:
        print("No existing vector store found. Creating new one...")
        # Load and process the document
        chunks = load_and_split_document(SPEECH_FILE)
        # Create vector store
        vectorstore = create_vector_store(chunks)
    
    # Set up QA chain
    rag_chain, retriever = setup_qa_chain(vectorstore)
    
    return rag_chain, retriever

def ask_question(rag_chain, retriever, question):
    """
    Ask a question and get an answer from the system
    
    Args:
        rag_chain: RAG chain
        retriever: Document retriever
        question (str): User's question
    
    Returns:
        tuple: (answer, source_documents)
    """
    print(f"\nQuestion: {question}")
    print("Retrieving relevant context and generating answer...\n")
    
    # Get relevant documents using invoke method
    source_docs = retriever.invoke(question)
    
    # Generate answer
    answer = rag_chain.invoke(question)
    
    return answer, source_docs

def main():
    """
    Main function to run the Q&A system
    """
    print("=" * 60)
    print("AmbedkarGPT - Dr. B.R. Ambedkar Speech Q&A System")
    print("=" * 60)
    print()
    
    # Check if speech file exists
    if not os.path.exists(SPEECH_FILE):
        print(f"Error: {SPEECH_FILE} not found!")
        print("Please ensure speech.txt is in the same directory as this script.")
        return
    
    try:
        # Initialize the system
        rag_chain, retriever = initialize_system()
        
        print("=" * 60)
        print("System ready! You can now ask questions about the speech.")
        print("Type 'quit', 'exit', or 'q' to exit the program.")
        print("=" * 60)
        
        # Interactive Q&A loop
        while True:
            print("\n" + "-" * 60)
            user_question = input("\nYour Question: ").strip()
            
            # Check for exit commands
            if user_question.lower() in ['quit', 'exit', 'q', '']:
                print("\nThank you for using AmbedkarGPT. Goodbye!")
                break
            
            # Get answer
            answer, source_docs = ask_question(rag_chain, retriever, user_question)
            
            # Display answer
            print("-" * 60)
            print(f"Answer: {answer}")
            print("-" * 60)
            
            # Optionally show source documents
            if source_docs:
                print(f"\n[Retrieved {len(source_docs)} relevant chunks]")
    
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nPlease ensure:")
        print("1. Ollama is installed and running")
        print("2. Mistral model is pulled (ollama pull mistral)")
        print("3. All required packages are installed (pip install -r requirements.txt)")

if __name__ == "__main__":
=======
"""
AmbedkarGPT - Q&A System using RAG Pipeline
A command-line application that answers questions based on Dr. B.R. Ambedkar's speech
using LangChain, ChromaDB, HuggingFace Embeddings, and Ollama Mistral 7B
"""

import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Constants
SPEECH_FILE = "speech.txt"
PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral"

def load_and_split_document(file_path):
    """
    Load the text file and split it into manageable chunks
    
    Args:
        file_path (str): Path to the speech text file
    
    Returns:
        list: List of document chunks
    """
    print(f"Loading document from {file_path}...")
    
    # Load the text file
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    
    # Split the text into chunks
    # Using smaller chunk size for better retrieval on short document
    text_splitter = CharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separator="\n"
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"Document split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks):
    """
    Create embeddings and store them in ChromaDB vector store
    
    Args:
        chunks (list): List of document chunks
    
    Returns:
        Chroma: Vector store instance
    """
    print("Creating embeddings using HuggingFace model...")
    
    # Initialize HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create and persist the vector store
    print("Storing embeddings in ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    vectorstore.persist()
    
    print(f"Vector store created and persisted to {PERSIST_DIRECTORY}")
    return vectorstore

def load_existing_vector_store():
    """
    Load an existing vector store from disk
    
    Returns:
        Chroma: Vector store instance
    """
    print("Loading existing vector store...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    
    return vectorstore

def setup_qa_chain(vectorstore):
    """
    Set up the RetrievalQA chain with Ollama LLM
    
    Args:
        vectorstore (Chroma): Vector store instance
    
    Returns:
        RetrievalQA: Question-answering chain
    """
    print("Setting up QA chain with Ollama Mistral 7B...")
    
    # Initialize Ollama LLM
    llm = Ollama(
        model=LLM_MODEL,
        temperature=0.3  # Lower temperature for more focused answers
    )
    
    # Create a custom prompt template
    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
Keep your answer concise and directly related to the question.

Context: {context}

Question: {question}

Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    print("QA chain ready!\n")
    return qa_chain

def initialize_system():
    """
    Initialize the RAG system by loading or creating the vector store
    
    Returns:
        RetrievalQA: Question-answering chain
    """
    # Check if vector store already exists
    if os.path.exists(PERSIST_DIRECTORY):
        print("Existing vector store found.")
        vectorstore = load_existing_vector_store()
    else:
        print("No existing vector store found. Creating new one...")
        # Load and process the document
        chunks = load_and_split_document(SPEECH_FILE)
        # Create vector store
        vectorstore = create_vector_store(chunks)
    
    # Set up QA chain
    qa_chain = setup_qa_chain(vectorstore)
    
    return qa_chain

def ask_question(qa_chain, question):
    """
    Ask a question and get an answer from the system
    
    Args:
        qa_chain (RetrievalQA): Question-answering chain
        question (str): User's question
    
    Returns:
        dict: Result containing answer and source documents
    """
    print(f"\nQuestion: {question}")
    print("Retrieving relevant context and generating answer...\n")
    
    result = qa_chain({"query": question})
    
    return result

def main():
    """
    Main function to run the Q&A system
    """
    print("=" * 60)
    print("AmbedkarGPT - Dr. B.R. Ambedkar Speech Q&A System")
    print("=" * 60)
    print()
    
    # Check if speech file exists
    if not os.path.exists(SPEECH_FILE):
        print(f"Error: {SPEECH_FILE} not found!")
        print("Please ensure speech.txt is in the same directory as this script.")
        return
    
    try:
        # Initialize the system
        qa_chain = initialize_system()
        
        print("=" * 60)
        print("System ready! You can now ask questions about the speech.")
        print("Type 'quit', 'exit', or 'q' to exit the program.")
        print("=" * 60)
        
        # Interactive Q&A loop
        while True:
            print("\n" + "-" * 60)
            user_question = input("\nYour Question: ").strip()
            
            # Check for exit commands
            if user_question.lower() in ['quit', 'exit', 'q', '']:
                print("\nThank you for using AmbedkarGPT. Goodbye!")
                break
            
            # Get answer
            result = ask_question(qa_chain, user_question)
            
            # Display answer
            print("-" * 60)
            print(f"Answer: {result['result']}")
            print("-" * 60)
            
            # Optionally show source documents
            if result.get('source_documents'):
                print(f"\n[Retrieved {len(result['source_documents'])} relevant chunks]")
    
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nPlease ensure:")
        print("1. Ollama is installed and running")
        print("2. Mistral model is pulled (ollama pull mistral)")
        print("3. All required packages are installed (pip install -r requirements.txt)")

if __name__ == "__main__":
>>>>>>> bd712ff86a74ca069b5a9866c9d6494c52558e43
    main()