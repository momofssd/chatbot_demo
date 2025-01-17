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
from datetime import datetime, timedelta

def parse_date(date_str):
    """Convert date string to datetime object"""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        try:
            return datetime.strptime(date_str, "%m/%d/%Y")
        except ValueError:
            return None

today_date = datetime.now().strftime("%Y-%m-%d")
today_datetime = parse_date(today_date)

# Constants
CHUNK_SIZE = 500
CHUNK_OVERLAP = 0


# Sidebar: API Key Input and Validation

st.sidebar.title("API Configuration")
if "api_key" not in st.session_state:
    st.session_state.api_key = None

api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password", placeholder="sk-...", help="Your OpenAI API key is required to access the chatbot functionality.")

if api_key:
    try:
        # Test the API key with a simple HTTP request
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.openai.com/v1/models", headers=headers)

        if response.status_code == 200:
            st.sidebar.success("API key is valid!")
            st.session_state.api_key = api_key
        else:
            error_message = response.json().get('error', {}).get('message', 'Unknown error')
            st.sidebar.error(f"Invalid API key. Error: {error_message}")
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"An error occurred while validating the API key: {str(e)}")
else:
    st.sidebar.info("Please enter your OpenAI API key to proceed.")



# Proceed only if API key is valid
if st.session_state.api_key:
    # Function to load vector store from train.txt
    def load_vector_store(file_path):
        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
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
        text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        return text_splitter.split_documents(documents)

    # Load documents and create vector store
    train_docs = load_vector_store("./train.txt")
    orders_docs = load_orders_vector_store("./orders.json")
    # Weight train_docs 3 times more
    all_docs = train_docs * 3 + orders_docs
    # Initialize embeddings with the API key
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state.api_key)
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})


    # Initialize memory and store it in session_state
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)
        st.session_state.memory.save_context(
            {"input": f"Today's date is {today_date}. You are a highly professional and empathetic customer service assistant."},
            {"output": (
                "You are a highly professional customer service representative with extensive experience. Your role includes:"
                "\n1. **General Information Provider**:"
                "\n   - Provide accurate information about ALL company services and facilities"
                "\n   - Share details about distribution centers, business hours, and policies"
                "\n   - Answer questions about shipping, returns, and general inquiries"
                "\n2. **Order Management Specialist**:"
                "\n   - Handle order-related inquiries when specifically asked"
                "\n   - Process returns and cancellations when requested"
                "\n   - Track shipments and delivery status"
                "\n3. **Professional Communication**:"
                "\n   - Use clear, concise language"
                "\n   - Stay focused on the customer's specific question"
                "\n   - Provide complete answers without assuming related needs"
                f"\n\nCRITICAL DATE AWARENESS - Today's date is: {today_date}"
                "\nALWAYS use this date for ALL eligibility calculations:"
                "\n- Returns: MUST check if within 5 days after delivery date"
                "\n- Cancellations: MUST check if within 5 days after purchase date"
                "\n- BEFORE providing ANY eligibility information:"
                f"\n  1. Compare against today's date ({today_date})"
                "\n  2. Calculate exact days difference"
                "\n  3. Clearly state if eligible or not based on date comparison"
                "\n- For delivered orders: Count days from delivery date to today"
                "\n- For pending orders: Count days from purchase date to today"
                "\n\nAdditional Policies:"
                "\n- Shipping: 3-5 business days standard delivery"
                "\n- Refunds: Processed within 5-7 business days"
                "\n\nOrder Handling Requirements:"
                "\n1. ALWAYS verify order details in orders.json"
                "\n2. For EVERY order-related query:"
                f"\n   - First state today's date: {today_date}"
                "\n   - Then state relevant order dates (purchase/delivery)"
                "\n   - Calculate and show days difference"
                "\n   - Clearly explain eligibility based on date comparison"
                "\n3. For duplicate orders, request specific order numbers"
                "\n\nResponse Structure:"
                "\n1. Acknowledge the customer's inquiry"
                "\n2. Provide relevant information/solution"
                "\n3. Add any necessary next steps or instructions"
                "\n4. Close with a professional, helpful statement"
                "\n\nMaintain consistent professionalism and focus on customer satisfaction throughout all interactions."
            )}
        )


    memory = st.session_state.memory



    # Initialize the retrieval QA chain
    retrieval_qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o",openai_api_key=st.session_state.api_key),
        retriever=retriever,
        memory=memory,
        
    )


    async def get_response(message):
        try:
            # Prepare context based on message content
            if any(keyword in message.lower() for keyword in ['order', 'return', 'cancel', 'delivery']):
                context = (
                    f"IMPORTANT DATE INFORMATION:\n"
                    f"- Today's date: {today_date}\n"
                    f"- All eligibility checks MUST use {today_date} as the reference date\n"
                    f"- When checking returns/cancellations:\n"
                    f"  1. First state today's date ({today_date})\n"
                    f"  2. Then state the relevant order dates\n"
                    f"  3. Calculate and show the exact days difference\n"
                    f"  4. Clearly explain if eligible based on the date difference\n\n"
                )
            else:
                # Check if it's a distribution center query
                if any(keyword in message.lower() for keyword in ['distribution', 'center', 'location', 'warehouse']):
                    context = (
                        "DISTRIBUTION CENTER INFORMATION GUIDELINES:\n"
                        "1. Provide complete address and contact details\n"
                        "2. Include business hours if available\n"
                        "3. List any specific services or capabilities\n"
                        "4. Do not discuss order processing unless specifically asked\n\n"
                    )
                else:
                    context = (
                        "RESPONSE GUIDELINES:\n"
                        "1. Focus specifically on the customer's question\n"
                        "2. Provide complete information from the knowledge base\n"
                        "3. Do not redirect to order-related topics unless specifically asked\n"
                        "4. Use clear, factual responses\n\n"
                    )
            
            date_context = f"{context}User message: {message}"
            response = await retrieval_qa_chain.arun(date_context)
            
            # Ensure date is mentioned in response if it's an order-related query
            if any(keyword in message.lower() for keyword in ['order', 'return', 'cancel', 'delivery']):
                if today_date not in response:
                    response = f"As of today ({today_date}): \n\n{response}"
            
            return response
        except Exception as e:
            return f"An error occurred: {e}"


    st.title("Professional Customer Service Assistant")
    st.subheader("How may I assist you today?")
    
    # Add helpful information
    with st.expander("💡 Quick Help"):
        st.markdown("""
        I can help you with:
        - Company Information & Facilities
          • Distribution centers
          • Business hours
          • Contact information
        - Order Management
          • Order status
          • Returns and cancellations
          • Shipping information
        - General Support
          • Policies and procedures
          • Services and programs
          • Other inquiries
        """)


    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    user_message = st.text_input("Your Message:", key="chat_input", help="Type your message here and press Enter to receive a response from the chatbot.")
    if user_message:
        st.session_state.chat_history.append({"role": "user", "message": user_message})
        with st.spinner("Generating response..."):
            response = asyncio.run(get_response(user_message))
            # Debug: Print updated memory state
            # print("Updated Memory State:")
            # print(memory.chat_memory.messages)
        st.session_state.chat_history.append({"role": "bot", "message": response})


    with st.container():
        st.markdown(
            """
            <style>
            .chat-container {
                height: 500px;
                overflow-y: auto;
                padding: 15px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: #ffffff;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .chat-container .user-message {
                background-color: #f0f4f8;
                color: #2c3e50;
                padding: 10px 15px;
                border-radius: 15px;
                margin: 10px 0;
                max-width: 80%;
                margin-left: auto;
                word-wrap: break-word;
            }
            .chat-container .bot-message {
                background-color: #e8f4ff;
                color: #34495e;
                padding: 10px 15px;
                border-radius: 15px;
                margin: 10px 0;
                max-width: 80%;
                word-wrap: break-word;
            }
            .chat-container strong {
                color: #1a73e8;
                font-weight: 600;
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
