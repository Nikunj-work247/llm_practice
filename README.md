# Customer Support Chatbot

This is a customer support chatbot built using Python, LangChain, and the Google Gemini API.

## Features Included

- **Caching**: Implemented using `langchain_community.cache.SQLiteCache`. If a user asks the exact same question again, the bot retrieves the answer from a local SQLite database (`.langchain.db`) instead of making a redundant and costly API call.
- **Memory Management**: Implemented using `ConversationBufferMemory`. The chatbot retains context of the conversation, allowing follow-up questions to be understood within the scope of previous messages.
- **Google Gemini Integration**: Uses `langchain-google-genai` and `ChatGoogleGenerativeAI` to leverage the Gemini 1.5 Flash model for quick, smart responses.

## Setup & Running

A virtual environment has been created and all dependencies installed.

To run the chatbot:
```bash
cd /home/lap-62/llm_practice/chatbot
source venv/bin/activate
python chatbot.py
```
