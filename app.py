from flask import Flask, render_template, jsonify, request
from src.helper import download_huggingface_embeddings
from src.prompt import system_prompt
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

app = Flask(__name__)

# --- Build RAG Pipeline ---
print("Loading embeddings model...")
embeddings = download_huggingface_embeddings()

index_name = "textbot"

# Connect to the existing Pinecone index (already populated via store_index.py)
print("Connecting to Pinecone vector store...")
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# LLM
llm = ChatGroq(temperature=0.4, model_name="llama-3.3-70b-versatile")

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# RAG Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("✓ Medical chatbot RAG pipeline ready.")


# --- Routes ---
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return jsonify({"answer": "Please enter a question."})

    print(f"User: {msg}")
    response = rag_chain.invoke({"input": msg})
    answer = response.get("answer", "I'm sorry, I couldn't find an answer.")
    print(f"Bot: {answer}")

    return str(answer)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)