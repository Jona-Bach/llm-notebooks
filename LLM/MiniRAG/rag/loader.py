from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_documents(folder_path = "../data"):

    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            path = os.path.join(folder_path,filename)
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()
            all_docs.extend(docs)
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    return split_docs