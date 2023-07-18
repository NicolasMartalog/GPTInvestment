from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
import os
import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# API key
os.environ['OPENAI_API_KEY'] = 'your-api-key'

# Creating an OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)
embeddings = OpenAIEmbeddings()

# Create and load PDF Loader seperatly and then use that for pages seperatly if you only want info for one bank
RBCloader = PyPDFLoader('RBCReport.pdf')
BMOloader = PyPDFLoader('BMOReport.pdf')
Scotialoader = PyPDFLoader('ScotiaBankReport.pdf')
TDloader = PyPDFLoader('TDReport.pdf')  

loader = DirectoryLoader('./', glob='./*.pdf', loader_cls=PyPDFLoader)

pages = loader.load_and_split()
# Load documents into vector database
store = Chroma.from_documents(pages, embeddings, collection_name='RBCReport')

vectorstore_info = VectorStoreInfo(
    name="annual_banking_report",
    description="a banking annual report as a pdf",
    vectorstore=store
) 

# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)
st.title('GPT Banking Helper')
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = agent_executor.run(prompt)
    st.write(response)

    with st.expander('Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt) 
        st.write(search[0][0].page_content) 