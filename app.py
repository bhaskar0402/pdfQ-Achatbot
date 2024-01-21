from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from datasets import load_dataset
import cassio
import dotenv
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
from typing_extensions import Concatenate
from langchain.text_splitter import CharacterTextSplitter


load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.getenv('ASTRA_DB_APPLICATION_TOKEN') # enter the "AstraCS:..." string found in in your Token JSON file
ASTRA_DB_ID = "3b39c7e9-1f87-4ddf-882d-0ab790dff010" 

cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

llm = OpenAI(openai_api_key= os.getenv('OPENAI_API_KEY'))
embedding = OpenAIEmbeddings(openai_api_key= os.getenv('OPENAI_API_KEY'))


astra_vector_store = Cassandra(
    embedding=embedding,
    table_name="QApdfdemo",
    session=None,
    keyspace=None,
)
def read_text_pdf(uploaded_file):
    pdfreader = PdfReader(uploaded_file)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

# We need to split the text using Character Text Split such that it sshould not increse token size
def create_chunk(raw_text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 800,
        chunk_overlap  = 200,
        length_function = len,
        )
    texts = text_splitter.split_text(raw_text)
    return texts

def astra_store(texts):
    astra_vector_store.add_texts(texts[:50])
    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)
    return astra_vector_index

st.set_page_config(page_title="PDF Q-A bot")
st.header("PDF Q-A bot")
#input_text=st.text_area("Job Description: ",key="input")
uploaded_file=st.file_uploader("Upload your (PDF)...",type=["pdf"])


if uploaded_file is not None:
    st.write("PDF Uploaded Successfully")


input_text=st.text_area("ask question: ",key="query_text")
query_text= input_text.strip() 
submit1 = st.button('answer')
submit2 = st.button('clear data base')

if submit1:
    if uploaded_file is not None:
        raw_text=read_text_pdf(uploaded_file)
        text=create_chunk(raw_text)
        astra_vector_index=astra_store(text)
        answer=astra_vector_index.query(query_text , llm=llm).strip()
        st.subheader("The Repsonse is")
        st.write(answer)
    else:
        st.write("Please uplaod the pdf")

if submit2:
        astra_vector_store.delete_collection()
        st.write("data base all clear")



