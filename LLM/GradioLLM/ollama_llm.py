from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import gradio as gr


def chat(Eingabe):
    template = """Question: {question}

    Short and useful answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    model = OllamaLLM(model="gemma:2b")

    chain = prompt | model
    return chain.invoke({"question": {Eingabe}})

def gradio():

    input = gr.Textbox(label="Frage eingeben", placeholder="Schreibe deine Frage hier...")
    output_text = gr.Textbox(label="Antwort", interactive=False)
    gr.Interface(fn=chat, inputs=input, outputs=output_text).launch(share=False, inbrowser=True)

if __name__ == '__main__':
    gradio()