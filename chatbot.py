import os
from dotenv import load_dotenv

# Langchain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain
from langchain_core.prompts import PromptTemplate

# Load environment variables (like GEMINI_API_KEY)
load_dotenv(dotenv_path='../.env')

# 1. Enable Caching
# This will cache identical queries to a local SQLite database and avoid hitting the API.
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

def create_chatbot():
    # 2. Initialize the primary LLM (Nvidia)
    primary_llm = ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct",
        temperature=0.3,
        max_tokens=1024,
        api_key=os.environ.get("Nvidia_API_KEY")
    )

    # Initialize the fallback LLM (Google Gemini)
    # Ensure GEMINI_API_KEY is in your environment variables
    fallback_llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash", 
        temperature=0.3, # Low temperature for more deterministic/factual answers
        max_tokens=1024
    )

    # Configure the LLM with fallbacks to automatically shift API when limits are reached
    llm = primary_llm.with_fallbacks([fallback_llm])

    # 3. Setup Memory Management
    # ConversationBufferMemory stores the history of the conversation to maintain context.
    # We can also use ConversationSummaryMemory if conversations get very long to optimize token usage.
    memory = ConversationBufferMemory()

    # 4. Define a custom Prompt Template (Optional but recommended)
    # This guides the model's persona and behavior.
    template = """You are a helpful customer support chatbot. You answer customer queries accurately and concisely.
    
    Current conversation:
    {history}
    Customer: {input}
    Chatbot:"""
    
    PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=template
    )

    # 5. Create the Conversation Chain
    # This links the LLM, the Memory, and the Prompt Template together.
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=PROMPT,
        verbose=True 
    )
    
    return conversation

def chat():
    print("Welcome to Customer Support! (Type 'quit' or 'exit' to end)")
    chatbot_chain = create_chatbot()
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Chatbot: Goodbye! Have a great day.")
            break
            
        # The chain handles appending to memory and sending to the LLM
        response = chatbot_chain.predict(input=user_input)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chat()
