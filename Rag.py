from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
#Splitting the Text
from langchain.text_splitter import RecursiveCharacterTextSplitter


#Loading PDF
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
#streamlit

st.set_page_config(page_title = "Pdf QA RAG based Chatbot",
                    layout = 'centered',
                    initial_sidebar_state = 'collapsed')
st.header('Self Made')

pdf = st.file_uploader('Upload your pdf here', type = 'pdf')
if pdf is not None:
    docs = []
    reader = PdfReader(pdf)
    i = 1
    for page in reader.pages:
        doc = docs.append(Document(page_content=page.extract_text(), metadata={'page':i}))
        i += 1


    if doc:
        splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 300)
        document = splitter.split_documents(docs)


        #Embedding and Storing
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.vectorstores import FAISS
        vector = FAISS.from_documents(document,OllamaEmbeddings())

        #LLM model
        from langchain_community.llms import Ollama
        llm = Ollama(model = 'llama2')

        #Prompting
        from langchain_core.prompts import ChatPromptTemplate
        from langchain import hub
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system","You are a helpful assistant{context}. Please response to the user queries"),
                ("user","Question:{input}")
            ]
        )

        #Retriever
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        doc_chain = create_stuff_documents_chain(llm = llm,prompt = prompt)
        retriever = vector.as_retriever()
        chain = create_retrieval_chain(retriever,doc_chain)

        question = st.text_input('Write your Query')
        result = chain.invoke({"input":question})

        generate = st.button('Response')
        if generate:
            print(result)
    else:
        st.error("Not a valid file")


