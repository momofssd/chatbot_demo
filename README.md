# Chat Bot Demo with Prompt Engineering

This is a Streamlit application that demonstrates a chatbot with prompt engineering using OpenAI's language model. The application allows users to interact with a chatbot that retrieves information from preloaded documents and provides contextually relevant responses.

## Features

- **API Key Configuration**: Securely input and validate your OpenAI API key.
- **Document Retrieval**: Load and process documents from `train.txt` and `orders.json` for information retrieval.
- **Memory Management**: Maintain conversation context with `ConversationBufferMemory`.
- **Interactive Chat Interface**: Input messages and receive responses with a user-friendly interface.

## Setup Instructions

1. **Install Dependencies**: Ensure you have Python and Streamlit installed. You can install the required packages using:
   ```bash
   pip install streamlit langchain faiss-cpu openai
   ```

2. **Run the Application**: Start the Streamlit app by running:
   ```bash
   streamlit run app2.py
   ```

3. **Enter API Key**: Enter your OpenAI API key in the sidebar to enable the chatbot functionality.

## Usage

- **Input Messages**: Type your message in the input box and press Enter.
- **View Responses**: The chatbot will generate a response based on the input and display it in the chat history.

## Dependencies

- `streamlit`
- `langchain`
- `faiss-cpu`
- `openai`
- `requests`
- `json`

## Notes

- Ensure your OpenAI API key is valid to use the chatbot.
- The application uses a retrieval-based QA chain to provide accurate and contextually relevant responses.

## License

This project is licensed under the MIT License.
