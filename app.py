from flask import Flask, render_template, request

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.helper import get_retriever
from src.prompt import system_prompt


app = Flask(__name__)


# Connect to existing Pinecone index
retriever = get_retriever(
    index_name="medical-chatbot",
    k=3
)


# Local Ollama LLM
chatModel = ChatOllama(
    model="llama3.2:3b",
    temperature=0.3
)


# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


# RAG chain
question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg")

    if not msg:
        return "Please enter a question.", 400

    print("User question:", msg)

    try:
        response = rag_chain.invoke(
            {
                "input": msg
            }
        )

        answer = response["answer"]

        print("Bot response:", answer)

        return str(answer)

    except Exception as e:
        print("Error:", str(e))
        return "Something went wrong while generating the answer. Please try again.", 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )