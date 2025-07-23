from rag.loader import load_documents
from rag.embedder import build_vectorstore
from rag.qa import build_qa_chain

def logic():
    documents = load_documents("data/")
    vectorstore = build_vectorstore(documents=documents)
    qa_chain = build_qa_chain(vectorstore=vectorstore)

    query = "Was mag Jonathan und was mag er nicht? Wo wollen er und Lisa mal hin?"
    result = qa_chain.invoke({"input": query})

    print("Antwort:", result["answer"])


if __name__ == "__main__":
    logic()