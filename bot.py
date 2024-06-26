from langchain_together import Together
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import os
import re
api_key=os.environ.get('TOGETHER_AI_API_KEY')
llm=Together(together_api_key=api_key,
             model='meta-llama/Meta-Llama-3-70B',
             temperature=0.5,
             max_tokens=512,
             top_p=0.9,   )
dfrm=pd.read_csv('a-b-c-d-e-f-g-h-i-j-k-l-m.csv')
agent=create_pandas_dataframe_agent(llm=llm,df=dfrm,verbose=False,return_intermediate_steps=True,max_iterations=10,max_execution_time=60)
answer=agent.invoke('What medicine is used to treat asthma?')
intermediate_output=answer['intermediate_steps'][7][1]
print(intermediate_output)