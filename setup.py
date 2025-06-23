#PyPDFLoader: reads pdf 
#DirectoryLoader : reads all pdf from 1 folder 
#RecursiveCharacterTextSplitter = splits large text into chunks
#HuggingFaceEmbeddings: model that converts text to vectors 
#FAISS: library that stores vectors and retreives fast based on semantic similarity of chunks 

print(" Script started!")
#loads api keys forn .env file
from dotenv import load_dotenv
load_dotenv()

import os
#prints api key for verification
print(os.getenv("HUGGINGFACEHUB_API_TOKEN"))

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#mentioning the path to the folder having all the pdfs
DATA_PATH = "Pharma_Regulatory_Docs/"

#functions access the folder , reads all pdf files and 
# saves as document having all the text
def load_pdf_files(Pharma_Regulatory_Docs):
    loader = DirectoryLoader(
        DATA_PATH,
        glob='*.pdf',  
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

#prints the total no.of pages from all the pdfs combined
documents = load_pdf_files(Pharma_Regulatory_Docs=DATA_PATH)
print("Length of the PDF pages:", len(documents))

#function to create chunks
#each chunk has 500 characters 
#overlap must be of 50 characters to maintain context 
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
    text_chunks= text_splitter.split_documents(extracted_data)
    return text_chunks

#prints the total no.of chunks made from content of all pdfs 
text_chunks= create_chunks(extracted_data=documents)
print("length of text chunks: ",len(text_chunks))

#function that loads the pretrained model 
#all-MiniLM-L6-v2: model that maps text to a dimention space used for clusters or semantic search
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model
#returns the model back and doesnt store it anywhere 

embedding_model=get_embedding_model()
#it calls it and it gets stored in a vairable 

#store embeddings in faiss (vector store)
#metion path of storing db
#give text chunks to embedding model to create embeddings
#save it in the path of the db stored
DB_FAISS_PATH = "vectorestore\db_faiss"
db=FAISS.from_documents(text_chunks,embedding_model)
db.save_local(DB_FAISS_PATH)




