from langchain_openai import AzureOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader,
    WebBaseLoader
)
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import ChatMessageHistory
import chromadb
import os
from typing import List, Dict
from dotenv import load_dotenv

class MultiSourceChatbot:
    def __init__(self):
        load_dotenv()
        
        # Create a persistent directory for Chroma
        self.persist_directory = "chroma_db"
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize Azure OpenAI
        self.llm = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY"),
            api_version='2024-02-15-preview',
            deployment_name='gpt-35-turbo'
        )
        
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
             api_key=os.getenv("AZURE_API_KEY"),
            api_version='2024-02-15-preview',
        #    deployment_name='text-embedding-3-large'  # Specify embedding model deployment
        )
        
        # Initialize Chroma client
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        
        self.chat_history = ChatMessageHistory()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=self.chat_history,
            return_messages=True
        )
        
        # Initialize vector store with persistence
        self.vector_store = Chroma(
            client=self.chroma_client,
            collection_name="document_collection",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
    def load_documents(self, sources: Dict[str, List[str]]) -> None:
        """Load documents with persistence"""
        documents = []
        
        for source_type, paths in sources.items():
            for path in paths:
                try:
                    if source_type == "text":
                        loader = TextLoader(path)
                        documents.extend(loader.load())
                    elif source_type == "pdf":
                        loader = PyPDFLoader(path)
                        documents.extend(loader.load())
                    elif source_type == "web":
                        loader = WebBaseLoader(path)
                        documents.extend(loader.load())
                except Exception as e:
                    print(f"Error loading {path}: {str(e)}")
                    continue

        if not documents:
            raise ValueError("No documents were successfully loaded")

        # Split documents
        splits = self.text_splitter.split_documents(documents)
        
        # Add documents to vector store
        try:
            self.vector_store.add_documents(splits)
            # Explicitly persist the database
            self.vector_store.persist()
            print(f"Successfully added {len(splits)} document chunks to vector store")
        except Exception as e:
            print(f"Error adding documents to vector store: {str(e)}")
            raise

    def setup_chain(self) -> None:
        """Initialize the conversation chain"""
        template = """You are a helpful assistant that can answer questions based on the provided context.
        Always try to base your answers on the context provided. If you can't find the answer in the context,
        say so clearly and try to provide a general response based on your knowledge.
        
        Context: {context}
        
        Chat History: {chat_history}
        
        Human: {question}
        Assistant:"""
        
        PROMPT = ChatPromptTemplate.from_template(template)
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 2}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            verbose=True
        )

    def chat(self, question: str) -> str:
        """Process a question with error handling"""
        if not hasattr(self, 'chain'):
            raise ValueError("Please setup the chain first using setup_chain()")
            
        try:
            response = self.chain.invoke({"question": question})
            return response["answer"]
        except Exception as e:
            error_msg = f"Error during chat: {str(e)}"
            print(error_msg)
            return error_msg

def test_chatbot():
    """Test function with proper cleanup"""
    try:
        # Initialize chatbot
        chatbot = MultiSourceChatbot()
        
        # Create test document
        with open("test.txt", "w") as f:
            f.write("This is a test document about John Doe. John is a software engineer with 5 years of experience.")
        
        # Load documents
        sources = {"text": ["test.txt"]}
        chatbot.load_documents(sources)
        
        # Setup chain
        chatbot.setup_chain()
        
        # Test queries
        queries = [
            "Who is John Doe?",
            "What is John's profession?",
            "How many years of experience does John have?"
        ]
        
        for query in queries:
            print(f"\nQ: {query}")
            response = chatbot.chat(query)
            print(f"A: {response}")
            
    except Exception as e:
        print(f"Test failed: {str(e)}")
        
    finally:
        # Cleanup test file
        if os.path.exists("test.txt"):
            os.remove("test.txt")

if __name__ == "__main__":
    test_chatbot()