from langchain_together import Together
import os
from flask import Flask, request, Response
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('TOGETHER_AI_API_KEY')
cohere_api_key=os.getenv('COHERE_API_KEY')
google_api_key=os.getenv('GOOGLE_API_KEY')
llm = Together(together_api_key=api_key,
               model='meta-llama/Meta-Llama-3-70B',
               temperature=0.0,
               max_tokens=512,
)
loader = CSVLoader('a-b-c-d-e-f-g-h-i-j-k-l-m.csv',  encoding="utf-8",csv_args={
        "delimiter": ","})

data = loader.load()
embeddings=GoogleGenerativeAIEmbeddings(google_api_key=google_api_key,model='models/text-embedding-004',task_type='clustering')
db=FAISS.from_documents(data,embedding=embeddings,)
retriever=db.as_retriever()
system_prompt = (
    "You are a helpful AI assistant. "
    "Make your response as accurate as possible. "
    "Provide only the medicine names and do not provide uses or dosage information. "
    "In order to query use data['Use'].contains()"
    "{context}"
)

prompt = PromptTemplate.from_template(template=system_prompt)
combine_docs_chain=create_stuff_documents_chain(llm=llm,prompt=prompt)
chain=create_retrieval_chain(retriever=retriever,combine_docs_chain=combine_docs_chain)
question = "Which medicine is used to treat diabetes?"

response=chain.invoke({"input":question})
print(response['answer'])