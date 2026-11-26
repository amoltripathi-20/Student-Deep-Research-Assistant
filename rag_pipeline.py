from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the embedding model once per app process."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(text: str):
    if not text or not text.strip():
        raise ValueError("Cannot create a vector store from empty text.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)

    if not chunks:
        raise ValueError("Text splitting produced no chunks.")

    return Chroma.from_texts(
        texts=chunks,
        embedding=get_embedding_model(),
    )


def retrieve_context(vector_db, query: str, k: int = 4) -> str:
    if not query or not query.strip():
        query = "Summarize the document"

    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    documents = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in documents)
