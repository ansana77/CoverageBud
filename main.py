import streamlit as st
from PyPDF2 import PdfReader
import uuid
import infer
from langchain.schema.document import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from htmltemplates import css, bot_template, user_template

def main():
    st.set_page_config(page_title = "Know your insurance better", page_icon =":books")
    st.write(css, unsafe_allow_html=True)

    st.header("Know your insurance better!")
    user_query = st.text_input("Ask a question about insurance plans: ")

    MAX_FILES = 2
    with st.sidebar:
        st.subheader("Your documents:")
        uploaded_pdfs = st.file_uploader("Upload your insurance plan brochures here and click on Process.", accept_multiple_files= True)

        if uploaded_pdfs:
            check_pdf_size(uploaded_pdfs, MAX_FILES)

        if st.button("Process"):
            with st.spinner("Processing"):
                #get the raw content from the pdf
                uuid_dict = {}
                raw_texts = []
                for pdf in uploaded_pdfs:
                    pdf_uuid = str(uuid.uuid4())
                    uuid_dict[pdf_uuid] = pdf.name
                    raw_text = get_pdf_text(pdf)
                    text_chunks = split_raw_text(raw_text, pdf_uuid)
                    raw_texts.extend(text_chunks)

                st.session_state.uuid_dict = uuid_dict

                # #create the vectorstore
                st.session_state.vectorstore = get_vectorstore(text_chunks)

                # # retrieve docs from rag
                st.session_state.retrieved_docs = infer.retrieve_chunks(st.session_state.vectorstore, uuid_dict, user_query)

                st.write("Documents processed you can now ask questions and press ask.")
                
    if st.button("Ask"):            
        if user_query and ("uuid_dict" and "retrieved_docs" in st.session_state):
            handle_user_input(user_query, st.session_state.uuid_dict, st.session_state.retrieved_docs)

def get_pdf_text(uploaded_pdf):
    text = ""
    pdf_reader = PdfReader(uploaded_pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_raw_text(raw_text, pdf_uuid):
    text_splitter = RecursiveCharacterTextSplitter(        
        chunk_size=1000,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,)
    raw_text_chunks = text_splitter.split_text(raw_text)

    chunks = []
    for text_chunk in raw_text_chunks:
        chunk = Document(page_content= text_chunk, metadata={"pdf_uuid": pdf_uuid})
        chunks.append(chunk)
    return chunks

def get_embedding_function():
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )
    return embeddings

def get_vectorstore(text_chunks):
    embeddings = get_embedding_function()
    text_content = [chunk.page_content for chunk in text_chunks]
    metadata = [chunk.metadata for chunk in text_chunks]
    vectorstore = FAISS.from_texts(texts=text_content, embedding= embeddings, metadatas=metadata)
    return vectorstore

def handle_user_input(user_query, uuid_dict, retrived_docs_rag):
    reply = infer.reply_generator(user_query, uuid_dict, retrived_docs_rag)
    st.write(user_template.replace(
                "{{MSG}}", reply.content), unsafe_allow_html=True)

def check_pdf_size(uploaded_pdfs, MAX_FILES):
    if len(uploaded_pdfs) > MAX_FILES:
        st.error(f"You can only upload up to {MAX_FILES} files.")
    else:
        st.success(f"{len(uploaded_pdfs)} file(s) uploaded successfully!")
    
    # Process the files here
    for pdf in uploaded_pdfs:
        # Process each PDF file
        st.write(f"Processing {pdf.name}")


if __name__ == "__main__":
    main()