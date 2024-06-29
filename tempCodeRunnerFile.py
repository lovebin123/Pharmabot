from langchain_together import Together
import os
from flask import Flask, request, Response
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_cohere import CohereEmbeddings
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('TOGETHER_AI_API_KEY')
cohere_api_key=os.getenv('COHERE_API_KEY')
llm = Together(together_api_key=api_key,
               model='meta-llama/Meta-Llama-3-70B',
               temperature=0.5,
               max_tokens=280,
)
loader = CSVLoader('a-b-c-d-e-f-g-h-i-j-k-l-m.csv',  encoding="utf-8")
data = loader.load()
splitter=RecursiveCharacterTextSplitter(chunk_size=50,chunk_overlap=0,separators=[','," "])
splittedDocs=splitter.split_documents(data)
embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key,model='embed-english-v3.0',)
db=FAISS.from_documents(splittedDocs,embedding=embeddings)
retriever=db.as_retriever()
system_prompt = (
    "You are a virtual pharmacist" 
    "You should answer when they ask about medicines"
    "You should only answer within the context"
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Give a precise answer and answer only the question asked."
    "\n\n"
    "{context}"
)
prompt = PromptTemplate.from_template(system_prompt)
combine_docs_chain=create_stuff_documents_chain(llm=llm,prompt=prompt)
chain=create_retrieval_chain(retriever=retriever,combine_docs_chain=combine_docs_chain)
question = "Which medicine is used to treat diabetes?"

response=chain.invoke({"input":question})
print(response['answer'])