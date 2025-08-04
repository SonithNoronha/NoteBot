import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit import sidebar
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

OpenAI_API_KEY = "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"

st.header("NoteBot")

with sidebar:
    st.title("My Notes")

    file = st.file_uploader("Upload your PDF Notes", type = "pdf")

if file is not None:
    my_pdf = PdfReader(file)
    text=""
    for page in my_pdf.pages:
        text += page.extract_text()
        #st.write(text)

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_text(text)
    #st.write(chunks)

    embeddings = OpenAIEmbeddings(api_key= OpenAI_API_KEY)

    vector_store = FAISS.from_texts(chunks, embeddings)

    user_query = st.text_input("Enter your query here")

    if user_query:
        matching_chunks = vector_store.similarity_search(user_query)

        llm = ChatOpenAI(
            api_key=OpenAI_API_KEY,
            max_tokens=300,
            temperature=0,
            model="gpt-3.5-turbo"
        )

        chain=load_qa_chain(llm, chain_type="stuff")
        output=chain.run(question=user_query, documents=matching_chunks)
        st.write(output)
