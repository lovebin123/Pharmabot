from langchain_together import Together
import os
import json
from flask import Flask, request, Response
from langchain.text_splitter import RecursiveJsonSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()
#Loading api keys
api_key = os.getenv('TOGETHER_AI_API_KEY') 
cohere_api_key=os.getenv('COHERE_API_KEY')
google_api_key=os.getenv('GOOGLE_API_KEY')
llm = Together(together_api_key=api_key,
               model='meta-llama/Meta-Llama-3-70B',
               temperature=0.0,
               max_tokens=512,
)
loader=open('Medicines.json',encoding='utf-8')
data=json.load(loader)
splitter=RecursiveJsonSplitter()
splitted_docs=splitter.create_documents(texts=data)
embeddings=GoogleGenerativeAIEmbeddings(google_api_key=google_api_key,model='models/text-embedding-004')
db=Chroma.from_documents(splitted_docs,embedding=embeddings)
retriever=db.as_retriever()
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
question = "Which medicines are used to treat dermatitis ?"
response=chain.invoke(question)
index=response.find('`')
if(index!=-1):
    print(response[:index])
else:
    print(response)