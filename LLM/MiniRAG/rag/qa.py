from langchain.chains.retrieval import create_retrieval_chain
from langchain_ollama.llms import OllamaLLM
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def build_qa_chain(vectorstore):
    model = OllamaLLM(model="gemma:2b")
    retriever = vectorstore.as_retriever()
    

    system_prompt = (
        "Nutze den gegebenen Kontext, um die Frage zu beantworten. "
        "Wenn du es nicht weißt, sag 'Ich weiß es nicht'. "
        "Maximal drei Sätze. Kontext: {context}"
    )


    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    doc_chain = create_stuff_documents_chain(model, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)
    return retrieval_chain