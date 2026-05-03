from flask import Flask, render_template, request, session

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.helper import get_retriever
from src.prompt import system_prompt


app = Flask(__name__)

app.secret_key = "medical-chatbot-secret-key"


# Simple in-memory session memory
chat_memory = {}


# Connect to existing Pinecone index
retriever = get_retriever(
    index_name="medical-chatbot",
    k=6
)


# Local Ollama LLM
chatModel = ChatOllama(
    model="llama3.2:3b",
    temperature=0.2
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)


rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


def get_session_id():
    if "session_id" not in session:
        session["session_id"] = "user_session"

    return session["session_id"]


def get_user_memory(session_id):
    if session_id not in chat_memory:
        chat_memory[session_id] = {
            "last_topic": None,
            "history": []
        }

    return chat_memory[session_id]


def is_follow_up_question(question):
    """
    Detects whether the current question depends on previous context.
    """
    question_lower = question.lower().strip()

    follow_up_terms = [
        "it",
        "its",
        "this",
        "that",
        "they",
        "them",
        "these",
        "those",
        "treatment",
        "symptoms",
        "causes",
        "how long",
        "how much time",
        "improve",
        "cure",
        "prevent",
        "side effects"
    ]

    return any(term in question_lower for term in follow_up_terms)


def extract_topic_from_question(question):
    """
    Basic medical topic extraction from direct questions.
    This avoids relying on the LLM for simple topic memory.
    """
    question_clean = question.strip().replace("?", "")

    patterns = [
        "what is ",
        "what are ",
        "explain ",
        "define ",
        "tell me about ",
        "treatment of ",
        "causes of ",
        "symptoms of ",
        "prevention of ",
    ]

    question_lower = question_clean.lower()

    for pattern in patterns:
        if pattern in question_lower:
            start_index = question_lower.find(pattern) + len(pattern)
            topic = question_clean[start_index:].strip()

            if topic:
                return topic.title()

    return None


def build_contextual_question(user_question, memory):
    """
    Rewrites follow-up questions using the last known medical topic.
    """
    last_topic = memory.get("last_topic")

    if last_topic and is_follow_up_question(user_question):
        contextual_question = f"For {last_topic}, {user_question}"
        return contextual_question

    return user_question


def update_memory(user_question, answer, memory):
    """
    Updates topic memory and conversation history.
    """
    extracted_topic = extract_topic_from_question(user_question)

    if extracted_topic:
        memory["last_topic"] = extracted_topic

    memory["history"].append(
        {
            "question": user_question,
            "answer": answer
        }
    )

    # Keep only last 5 turns
    memory["history"] = memory["history"][-5:]


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg")

    if not msg:
        return "Please enter a question.", 400

    session_id = get_session_id()
    memory = get_user_memory(session_id)

    contextual_question = build_contextual_question(msg, memory)

    print("Original question:", msg)
    print("Contextual question:", contextual_question)
    print("Current topic:", memory.get("last_topic"))

    try:
        response = rag_chain.invoke(
            {
                "input": contextual_question
            }
        )

        answer = response["answer"]

        update_memory(msg, answer, memory)

        print("Bot response:", answer)
        print("Updated topic:", memory.get("last_topic"))

        return str(answer)

    except Exception as e:
        print("Error:", str(e))
        return "Something went wrong while generating the answer. Please try again.", 500


@app.route("/clear", methods=["POST"])
def clear_chat():
    session_id = get_session_id()

    if session_id in chat_memory:
        chat_memory[session_id] = {
            "last_topic": None,
            "history": []
        }

    return "Chat memory cleared."


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )