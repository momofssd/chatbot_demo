import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationChain
import asyncio
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import json
import requests


# Sidebar: API Key Input and Validation

st.sidebar.title("API Configuration")
if "api_key" not in st.session_state:
    st.session_state.api_key = None

api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password", placeholder="sk-...")

if api_key:
    try:
        # Test the API key with a simple HTTP request
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.openai.com/v1/models", headers=headers)

        if response.status_code == 200:
            st.sidebar.success("API key is valid!")
            st.session_state.api_key = api_key
        else:
            st.sidebar.error(f"Invalid API key. Error: {response.json().get('error', {}).get('message', 'Unknown error')}")
    except Exception as e:
        st.sidebar.error(f"An error occurred: {str(e)}")
else:
    st.sidebar.info("Please enter your OpenAI API key to proceed.")



# Proceed only if API key is valid
if st.session_state.api_key:
    # Function to load vector store from train.txt
    def load_vector_store(file_path):
        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        return text_splitter.split_documents(documents)

# Function to process orders.json dynamically
    def load_orders_vector_store(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)

        documents = [
            Document(
                page_content="\n".join(f"{key}: {value}" for key, value in entry.items()),
                metadata=entry
            )
            for entry in data
        ]
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        return text_splitter.split_documents(documents)

    # Load documents and create vector store
    train_docs = load_vector_store("./train.txt")
    orders_docs = load_orders_vector_store("./orders.json")
    all_docs = train_docs * 3 + orders_docs  # Weight train_docs 3 times more
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.api_key)
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


    # Initialize memory and store it in session_state
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
        st.session_state.memory.save_context(
            {"input": "You are a highly professional and empathetic customer service assistant."},
            {"output": (
                "Your primary responsibility is to assist customers with their inquiries in a polite, concise, and helpful manner."
                " Always prioritize resolving issues using the knowledge from the weighted training documents."
                " Note: Confirmation status is not relevant to return or cancellation policies."
                " Reference the retrieved information wherever applicable and explain it clearly to the customer."
                " Ensure that responses are tailored to the customer's question, providing actionable steps or guidance."
                " For inquiries related to orders, utilize the preloaded order data from `orders.json` to verify details or eligibility."
                " When discussing returns or cancellations, strictly follow these policies:"
                "  - **Return Policy**: Returns are allowed within 5 days after delivery."
                "  - **Cancellation Policy**: Cancellations are allowed within 5 days after purchase (purchase date)."
                " If you cannot resolve a customer's issue directly, politely suggest they contact the appropriate department, such as order management, for further assistance."
                " Acknowledge and validate the customer's concerns to make them feel heard."
                " Structure your responses clearly, using bullet points or numbered steps for instructions when needed."
                " End your responses with an offer to assist further or clarify any additional questions."
                " Always maintain a professional tone and a focus on delivering exceptional customer service."
            )}
        )


    memory = st.session_state.memory



    retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o",openai_api_key=st.session_state.api_key),
        retriever=retriever,
        memory=memory,
        
    )


    async def get_response(message):
        try:
            return await retrieval_qa_chain.arun(message)
        except Exception as e:
            return f"An error occurred: {e}"


    st.title("Chat Bot Demo")
    st.subheader("With Prompt Engineering")


    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    user_message = st.text_input("Your Message:", key="chat_input")
    if user_message:
        st.session_state.chat_history.append({"role": "user", "message": user_message})
        with st.spinner("Generating response..."):
            response = asyncio.run(get_response(user_message))
            print("Updated Memory State:")
            print(memory.chat_memory.messages)
        st.session_state.chat_history.append({"role": "bot", "message": response})


    with st.container():
        st.markdown(
            """
            <style>
            .chat-container {
                height: 500px; /* Set a fixed height for the chat area */
                overflow-y: auto; /* Enable vertical scrolling */
                padding: 10px;
                border: 1px solid #ccc; /* Optional: Add a border */
                border-radius: 5px; /* Optional: Add rounded corners */
                background-color: #f9f9f9; /* Optional: Light background color */
            }
            .chat-container .user-message {
                color: blue;
            }
            .chat-container .bot-message {
                color: green;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


        chat_html = '<div class="chat-container">'
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                chat_html += f'<div class="user-message"><strong>You:</strong> {chat["message"]}</div>'
            else:
                chat_html += f'<div class="bot-message"><strong>Chatbot:</strong> {chat["message"]}</div>'
        chat_html += "</div>"


        st.markdown(chat_html, unsafe_allow_html=True)