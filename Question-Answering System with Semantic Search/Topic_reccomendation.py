import streamlit as st
import fitz  # PyMuPDF
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document  # Import the Document class

# Set API key for OpenAI
OPENAI_API_KEY = ''

# Set OpenAI LLM
llm_chat = ChatOpenAI(temperature=0.7, model='gpt-3.5-turbo', openai_api_key=OPENAI_API_KEY)

def extract_text_from_pdf(file) -> str:
    """Extracts text from a PDF file."""
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

def extract_text_from_txt(file) -> str:
    """Extracts text from a TXT file."""
    return file.read().decode("utf-8")

def generate_topic_questions(text: str) -> list:
    """Generates topic questions from the given text."""
    chain = load_qa_chain(llm_chat, chain_type="stuff")
    question_prompt = "Generate a list of topic questions based on the following text:"
    document = Document(page_content=text)
    response = chain.run(input_documents=[document], question=question_prompt)
    
    # Split the response into individual lines and remove any pre-existing numbering
    questions = response.split('\n')
    clean_questions = [q.lstrip('0123456789. ').strip() for q in questions if q.strip()]
    
    return clean_questions

# Streamlit app
st.title("File Upload and Topic Question Generator")

uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".pdf"):
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".txt"):
        text = extract_text_from_txt(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a PDF or TXT file.")
        st.stop()

    st.write("Extracted Text:")
    st.write(text[:5000])  # Display the first 5000 characters of the extracted text

    if st.button("Generate Topic Questions"):
        with st.spinner("Generating questions..."):
            questions = generate_topic_questions(text)
            if questions:
                st.write("Generated Topic Questions:")
                for idx, question in enumerate(questions, start=1):
                    st.write(f"{idx}. {question}")
            else:
                st.write("No questions were generated.")
