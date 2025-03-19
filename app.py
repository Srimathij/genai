import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

USER_CREDENTIALS = {"converse.cx": "Kgisl@12345"}

def authenticate(username, password):
    return USER_CREDENTIALS.get(username) == password

def login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.sidebar.subheader("Login")
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")
        login_button = st.sidebar.button("Login")

        if login_button:
            if authenticate(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.sidebar.success("Login successful!")
                st.experimental_rerun()
            else:
                st.sidebar.error("Invalid credentials")
    else:
        st.sidebar.success(f"Logged in as: {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.experimental_rerun()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = PromptTemplate(
        template="""
        You are an expert assistant that provides precise and highly accurate answers strictly based on the provided context.
        - If the answer is present, extract the most relevant details and present them clearly.
        - If the context is insufficient, respond with "The answer is not available in the provided document."
        - Do NOT attempt to answer questions beyond the given document.
        
        Context:
        {context}
        
        Question:
        {question}
        
        **Detailed Answer:**
        """,
        input_variables=["context", "question"],
    )
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-001", temperature=0.2)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt_template)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question, k=5)  # Fetch top 5 relevant chunks
    
    if not docs:
        st.write("**Response:** The answer is not available in the provided document.")
        return
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.markdown("**Response:**")
    for line in response["output_text"].split("\n"):
        if line.strip():
            st.markdown(f"- **{line.strip()}**")

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with Document LLM")
    
    user_question = st.text_input("Ask a question from the uploaded PDF files:")
    if user_question:
        user_input(user_question)
        
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF files and click 'Process'", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if raw_text.strip():
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete! You can now ask questions.")
                else:
                    st.error("No text could be extracted. Please upload a valid PDF.")

if __name__ == "__main__":
    main()
