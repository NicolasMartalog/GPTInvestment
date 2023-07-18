# This is for one document (RBC annual report)
from langchain.llms import OpenAI 
from langchain.embeddings import OpenAIEmbeddings
import os 
import streamlit as st  
from langchain.document_loaders import PyPDFLoader 
from langchain.vectorstores import Chroma

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent, 
    VectorStoreToolkit, 
    VectorStoreInfo
)

# Can sub this for other LLM providers
os.environ['OPENAI_API_KEY'] = 'your-api-key' 

# Creating instance of OpenAI LLM
llm = OpenAI(temperature=0.9, verbose=True) 
embeddings = OpenAIEmbeddings()

loader = PyPDFLoader('RBCReport.pdf')
# Split pages from pdf 
pages = loader.load_and_split() 
# load documents into vector database
store = Chroma.from_documents(pages, embeddings, collection_name='RBCReport.pdf') 

# Create vectorstore info object
vectorstore_info = VectorStoreInfo(
    name="annual_report", 
    description="a banking annual report as a pdf", 
    vectorstore=store
)

# Convert the document store inot a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info) 

# add the toolkit to an end-to-end LC
agent_exector = create_vectorstore_agent(
    llm=llm, 
    toolkit=toolkit, 
    verbose=True
)

st.title('GPT Banking Helper')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If enter is hit
if prompt:  
    # pass the prompt to the LLM
    #respone = llm(prompt)  

    # Swap out the raw llm for a document agent
    respone = agent_exector.run(prompt)

    ## Write it to the screen
    st.write(respone)  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt) 
        # Write out the first 
        st.write(search[0][0].page_content) 

