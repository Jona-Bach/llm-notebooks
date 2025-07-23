from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import os

def build_vectorstore(documents, dir = "./embeddings"):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


    if not os.path.exists(dir):
        os.mkdir(dir)
    elif os.path.exists(dir):
        print("[i] Lade bestehende FAISS-Datenbank...")
        return FAISS.load_local(dir, embeddings=embedding_model, allow_dangerous_deserialization=True)
    
    vectorstore = FAISS.from_documents(documents=documents,embedding=embedding_model)
    vectorstore.save_local(dir)
    return vectorstore
