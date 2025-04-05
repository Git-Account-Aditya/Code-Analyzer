from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationSummaryMemory
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
import streamlit as st
from streamlit_ace import st_ace

# Load .env variables
load_dotenv()

# Initialize LLM
groq = ChatGroq(model_name="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

# Memory for chat history
chat_memory = ConversationSummaryMemory(memory_key="chat_history", llm=groq)

# Prompt 1: Identify Language
prompt1 = PromptTemplate(
    template="Identify and output only the programming language used in this piece of code:\n\n{code}",
    input_variables=["code"]
)

# Prompt 2: Improvements
prompt2 = PromptTemplate(
    template="Based on this conversation history:\n{chat_history}\n\n"
             "Identify errors in this {language} code and provide ONLY THE IMPROVEMENTS, NOT THE CODE:\n\n{code}",
    input_variables=["chat_history", "language", "code"]
)

# Prompt 3: Final Fix + User Input
prompt3 = PromptTemplate(
    template="Based on this conversation history:\n{chat_history}\n\n"
             "Rectify the errors in this {language} code using these improvements:\n\n{improvements}\n\n"
             "Additionally, the user has requested the following changes or instructions:\n\n{user_input}\n\n"
             "Provide ONLY the corrected code:\n\n{code}",
    input_variables=["chat_history", "language", "code", "improvements", "user_input"]
)

# Parser
parser = StrOutputParser()

# ---- Streamlit UI ----
st.title("🧠 AI Code Helper")

# Code Editor
code = st_ace(
    value="print('Hello, Streamlit!')",
    language="python",
    theme="monokai",
    keybinding="vscode",
    height=300,
    auto_update=True
)

# User instruction input
user_input = st.text_area("💬 Enter any specific instruction (e.g., add error handling, use list comprehension):")

# Chains
chain1 = prompt1 | groq | parser

# Button click
if st.button("🔍 Analyze Code"):
    if not code:
        st.warning("Please enter some code to analyze.")
    else:
        # Step 1: Identify Language
        language = chain1.invoke({'code': code})

        # Step 2: Improvements
        improvements = parser.invoke(
            groq.invoke(prompt2.format(
                chat_history=chat_memory.load_memory_variables({})["chat_history"],
                language=language,
                code=code
            ))
        )

        st.subheader("🔧 Suggested Improvements")
        st.write(improvements)

        # Step 3: Final Code with User Input
        corrected_code = parser.invoke(
            groq.invoke(prompt3.format(
                chat_history=chat_memory.load_memory_variables({})["chat_history"],
                language=language,
                code=code,
                improvements=improvements,
                user_input=user_input
            ))
        )

        # Show corrected code
        st.subheader("✅ Final Corrected Code")
        st.code(corrected_code, language=language)

        # Save user request + AI output to memory
        if user_input:
            chat_memory.save_context(
                {"user": (user_input, code)},
                {"ai": corrected_code}
            )


test_code1 = '''
                if st.button("Submit"):
    if not text:
        st.warning("⚠️ Please upload a file first!")
    else if not question.strip():
        st.warning("⚠️ Please enter a question!")
    else if:
        # Invoke the model
        result = chain.invoke({"text": text, "question": question})
        st.write("🤖 **AI Response:**")
        st.write(result.content)
'''



