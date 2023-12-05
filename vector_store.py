from dotenv import load_dotenv, find_dotenv
import shutil
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import PyPDFLoader
import os
import openai

load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# Load PDF
loaders = [
    PyPDFLoader("docs/Case4.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)

# text_splitter = TokenTextSplitter(chunk_size=250, chunk_overlap=50)   #option for splitting based on tokens rather than characters

splits = text_splitter.split_documents(docs)

embedding = OpenAIEmbeddings()

# clear vector stores
persist_directory = 'docs/chroma/'
folder_path = r"docs/chroma/"  # enter your path
shutil.rmtree(folder_path)

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)
