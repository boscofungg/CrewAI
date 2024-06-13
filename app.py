import os

# First enter the  model of your choice
MODEL = "llama3"

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)


# The following line can be used to test the llm
# you may print out the model invoke for testing
model.invoke("Tell me a joke")



from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

chain = model | parser 
chain.invoke("Tell me a joke")


# Then we prompt the llm such that it follows our instruction, which is answer based on the study material
from langchain.prompts import PromptTemplate

template = """
Answer the question based on the context below. If you can't 
answer the question, reply "The answer to your question is not mentioned in the material you provided.".

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)
prompt.format(context="Here is some context", question="Here is a question")

chain = prompt | model | parser

# The following line can be used to test whether the llm is following our instructions
# You may print out the invoke for testing
chain.invoke({"context": "My parents named me Bosco", "question": "What's my brother's name'?"})

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("Random_Notes.pdf")
pages = loader.load_and_split()
pages

from langchain_community.vectorstores import DocArrayInMemorySearch

vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
retriever = vectorstore.as_retriever()

# The following line can be used to test the retriever
# You can print out the term for testing
retriever.invoke("cash flow")

from operator import itemgetter

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | parser
)
# questions = [
#     "What does Bosco and Victor has?",
#     "What will happen if we heat potassium?",
#     "What will happen if we heat copper?"
# ]

# for question in questions:
#     print(f"Question: {question}")
#     print(f"Answer: {chain.invoke({'question': question})}")
#     print()

# for s in chain.stream({"question": "What is the purpose of the course?"}):
#     print(s, end="", flush=True)

# from langchain_community.document_loaders import WebBaseLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# loader = WebBaseLoader("urllink")
# docs = loader.load()
# documents = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=200
# ).split_documents(docs)

# documents

# This is the streamlit code for the prototype
import streamlit as st
text_input = st.text_input(
        "Enter your questions 👇",
        placeholder=st.session_state.placeholder,
    )
questions = f'{text_input}'
for question in questions:
    st.write(f"Question: {question}")
    st.write(f"Answer: {chain.invoke({'question': question})}")
    st.write()