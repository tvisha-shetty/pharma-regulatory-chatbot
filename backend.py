import os
from typing import Optional, List
from dotenv import load_dotenv
'''os: For working with file paths and environment variables
 typing: Lets you specify input/output types (optional, list)
 load_dotenv: Loads secret API keys from a .env file'''
from langchain.document_loaders import PyMuPDFLoader as PyPDFLoader
#reads all content from pdf 
from langchain.text_splitter import RecursiveCharacterTextSplitter
#makes text chunks
from langchain.vectorstores import FAISS
#stores vectors for semantic search 
from langchain.chains import RetrievalQA
#connects faiss to llm
from langchain.llms.base import LLM
#own llm wrapper
from langchain.prompts import PromptTemplate
#custom prompts for bots behaviour
from langchain_huggingface import HuggingFaceEmbeddings
#creates embeddings from chunks
from together import Together
#python package is where mistral model is hosted

# Load API Key from .env
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    raise ValueError(" TOGETHER_API_KEY not found in .env file!")

# class that inherits from llm to work in langchain
class TogetherLLM(LLM):
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    max_tokens: int = 512 #output length of words+punc(1 token = 4 characters)
    temperature: float = 0.0  # factual to creative asnwer
    api_key: Optional[str] = None
#variable that stores the api key forwhenever its used  later
    @property
    #using custom llm known as together
    def _llm_type(self) -> str:
        return "together"

#function takes the question youre asking 
# when to stop upon seeing certain words
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
       
        client = Together(api_key=self.api_key)
        #creates a cconnection between the code and togeather ai service
        response = client.chat.completions.create(
            model=self.model_name, #same model
            messages=[{"role": "user", "content": prompt}], #message is from user with a prompt 
            max_tokens=self.max_tokens,#same tokens
            temperature=self.temperature,#same temp
            stop=stop #stop due to certain words
        )
        return response.choices[0].message.content.strip()
    #take the ai first answer , remove spaces and send it back

#function for multi pdfs chatbot
def run_multi_pdf_chatbot():
    #add input of file paths
    input_paths = input(" Enter full paths to your PDF files (comma-separated): ").split(',')
    #only accepts valid path / files ending with .pdf
    pdf_paths = [p.strip() for p in input_paths if p.strip().endswith(".pdf") and os.path.exists(p.strip())]

    if not pdf_paths:
        print(" No valid PDF files found.")
        return


#reading pdfs 
    print(" Loading PDFs...")
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())
        #returns content as documents and length of these documents
    print(f" Loaded {len(all_docs)} pages from {len(pdf_paths)} PDFs.")

    print(" Splitting into very small chunks for better focus...")
    #250 character in each chunk
    #40 character overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40)
    split_docs = splitter.split_documents(all_docs)

#converts chunks into vectors and stores in faiss
    print(" Creating vector store with better embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
#retreives top 15 most similar chunks
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 15})

    # Strict custom prompt to avoid hallucination
    custom_prompt = PromptTemplate.from_template(
        """
You are a highly accurate pharma regulatory assistant AI trained ONLY on the following context.  
You MUST NOT guess or hallucinate. If answer is not present, say exactly: 'Not found in context'.

Example:
Context: "The expiry date of Drug A is 2025-12-31."
Question: "What is the expiry date of Drug A?"
Answer: "2025-12-31"

Context:
{context}

Question: {question}
Answer:"""
    )


#creating the custom model class 
    llm = TogetherLLM(api_key=TOGETHER_API_KEY)#giving permission to access the model
    #connecting brain to documents 
    qa = RetrievalQA.from_chain_type(#retreival qa is question-answering chain.)
        llm=llm,#uses the brain we created 
        retriever=retriever,#uses the doc finder that searches the doc 
        chain_type="stuff",#sends all matched parts from pdf to brain
        chain_type_kwargs={"prompt": custom_prompt}#use the custom prompt to avoid guessing 
    )
#starts the loop for chatting
    print("\n Chat with your documents (type 'exit' to quit):")
    while True:
        #if user says exit ,quit,bye ,,thank you then it exits the chat .
        query = input("You: ")
        if any(x in query.lower() for x in ["exit", "quit", "bye", "thank you"]):
            print(" Exiting chatbot. Goodbye!")
            break
        print("Bot:", qa.run(query))
        """function of chatbot ends here """

if __name__ == "__main__":
    run_multi_pdf_chatbot()
    #if the file is being run directly and not imported then the run 
    #run_multi_pdf_chatbot()
