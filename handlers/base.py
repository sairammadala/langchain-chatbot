import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import UploadFile, HTTPException

# --- Pinecone v3 (>=5) + LangChain integrations ---
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# Embeddings / LLMs
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Chains / Vectorstore
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores.utils import DistanceStrategy  # optional (defaults to cosine)

# Loaders
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader

# Text splitters
from langchain.text_splitter import (
    TokenTextSplitter,
    TextSplitter,
    Tokenizer,
    Language,
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
    LatexTextSplitter,
    PythonCodeTextSplitter,
    KonlpyTextSplitter,
    SpacyTextSplitter,
    NLTKTextSplitter,
    SentenceTransformersTokenTextSplitter,
    ElementType,
    HeaderType,
    LineType,
    HTMLHeaderTextSplitter,
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
    CharacterTextSplitter,
)

# your helpers
from utils.alerts import alert_exception, alert_info

load_dotenv()


class BaseHandler:
    def __init__(
        self,
        chat_model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        **kwargs,
    ):
        # --- Pinecone (v3) env ---
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index = os.getenv("PINECONE_INDEX")
        self.pinecone_host = os.getenv("PINECONE_HOST")  # recommended with serverless

        if not self.pinecone_api_key:
            raise RuntimeError("Missing PINECONE_API_KEY")
        if not self.pinecone_index:
            raise RuntimeError("Missing PINECONE_INDEX")

        # Initialize Pinecone client (no environment)
        self.pc = Pinecone(api_key=self.pinecone_api_key)

        # LLM map (unchanged)
        self.llm_map = {
            "gpt-4": lambda: ChatOpenAI(model="gpt-4", temperature=temperature, openai_api_key=os.getenv("OPENAI_API_KEY")),
            "gpt-4-32k": lambda: ChatOpenAI(model="gpt-4-32k", temperature=temperature, openai_api_key=os.getenv("OPENAI_API_KEY")),
            "gpt-4-1106-preview": lambda: ChatOpenAI(model="gpt-4", temperature=temperature, openai_api_key=os.getenv("OPENAI_API_KEY")),
            "gpt-3.5-turbo-16k": lambda: ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=temperature, openai_api_key=os.getenv("OPENAI_API_KEY")),
            "gpt-3.5-turbo": lambda: ChatOpenAI(model="gpt-3.5-turbo", temperature=temperature, openai_api_key=os.getenv("OPENAI_API_KEY")),
            "claude-3-sonnet-20240229": lambda: ChatAnthropic(model_name="claude-3-sonnet-20240229", temperature=temperature, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")),
            "claude-3-opus-20240229": lambda: ChatAnthropic(model_name="claude-3-opus-20240229", temperature=temperature, anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")),
        }
        self.chat_model = chat_model

        # Embeddings
        if kwargs.get("embeddings_model") == "text-embedding-3-large":
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                model="text-embedding-3-large",
            )
            self.dimensions = 3072
        else:
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                model="text-embedding-3-small",
            )
            self.dimensions = 1536

    # ---------------------------
    # Document loading (unchanged)
    # ---------------------------
    def load_documents(self, files: List[UploadFile], namespace: Optional[str] = None) -> List[list]:
        documents = []
        loader_map = {
            "txt": TextLoader,
            "pdf": PyMuPDFLoader,
            "docx": Docx2txtLoader,
        }
        allowed_extensions = list(loader_map.keys())

        try:
            for file in files:
                ext = file.filename.split(".")[-1].lower()
                if ext not in allowed_extensions:
                    raise HTTPException(status_code=400, detail="File type not permitted")

                # Windows-safe temp handling
                with tempfile.NamedTemporaryFile(delete=False, prefix=file.filename + "___", suffix="." + ext) as tmp:
                    tmp_path = tmp.name
                    shutil.copyfileobj(file.file, tmp)

                try:
                    loader = loader_map[ext](tmp_path)
                    documents.append(loader.load())
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
        except Exception as e:
            alert_exception(e, "Error loading documents")
            raise HTTPException(status_code=500, detail=f"Error loading documents: {str(e)}")

        return documents

    # ---------------------------
    # Ingestion (new SDK usage)
    # ---------------------------
    def ingest_documents(self, documents: List[list], chunk_size: int = 1000, chunk_overlap: int = 100, **kwargs):
        """
        documents: list of loaded documents
        chunk_size / chunk_overlap: splitter params
        kwargs:
            split_method: one of the keys in splitter_map below
            namespace: optional string to segment your data
        """
        splitter_map = {
            "recursive": RecursiveCharacterTextSplitter,
            "token": TokenTextSplitter,
            "text": TextSplitter,
            "tokenizer": Tokenizer,
            "language": Language,
            "json": RecursiveJsonSplitter,
            "latex": LatexTextSplitter,
            "python": PythonCodeTextSplitter,
            "konlpy": KonlpyTextSplitter,
            "spacy": SpacyTextSplitter,
            "nltk": NLTKTextSplitter,
            "sentence_transformers": SentenceTransformersTokenTextSplitter,
            "element_type": ElementType,
            "header_type": HeaderType,
            "line_type": LineType,
            "html_header": HTMLHeaderTextSplitter,
            "markdown_header": MarkdownHeaderTextSplitter,
            "markdown": MarkdownTextSplitter,
            "character": CharacterTextSplitter,
        }

        split_method = kwargs.get("split_method", "recursive")
        splitter_cls = splitter_map[split_method]
        splitter = splitter_cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        ns = kwargs.get("namespace", None)
        alert_info(f"Ingesting {len(documents)} document(s)...\nParams: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, split_method={split_method}, namespace={ns}")

        try:
            for doc in documents:
                chunks = splitter.split_documents(doc)

                # Use the new LangChain vectorstore for Pinecone v3
                PineconeVectorStore.from_documents(
                    documents=chunks,
                    embedding=self.embeddings,
                    index_name=self.pinecone_index,
                    namespace=ns,
                    # host is optional if your SDK can resolve it by index name,
                    # but using host from console is most robust
                    host=self.pinecone_host,
                    pinecone_api_key=self.pinecone_api_key,
                    distance_strategy=DistanceStrategy.COSINE,  # matches your index metric
                )
        except Exception as e:
            alert_exception(e, "Error ingesting documents (dimension/host/index mismatch?)")
            raise HTTPException(status_code=500, detail=f"Error ingesting documents: {str(e)}")

    # ---------------------------
    # Chat / Retrieval (new SDK)
    # ---------------------------
    def chat(self, query: str, chat_history: List[str] = [], **kwargs):
        """
        kwargs:
            namespace: str
            search_kwargs: dict (e.g., {"k": 5})
        """
        ns = kwargs.get("namespace", None)
        search_kwargs = kwargs.get("search_kwargs", {"k": 5})

        try:
            # Build a Pinecone Index explicitly so we can use host reliably
            if self.pinecone_host:
                index = self.pc.Index(host=self.pinecone_host)
            else:
                index = self.pc.Index(self.pinecone_index)

            # Create vector store from the Index object (constructor, not classmethod)
            vectorstore = PineconeVectorStore(
                index=index,
                embedding=self.embeddings,
                namespace=ns,
                text_key="text",  # matches your ingestion key
            )
            retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

            # IMPORTANT: instantiate the LLM (call the lambda)
            llm = self.llm_map[self.chat_model]()

            bot = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
            )

            result = bot.invoke({"question": query, "chat_history": chat_history})
            return result

        except Exception as e:
            alert_exception(e, "Error chatting with Pinecone-backed retriever")
            raise HTTPException(status_code=500, detail=f"Error chatting: {str(e)}")