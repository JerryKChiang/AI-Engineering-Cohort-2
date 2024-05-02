# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

# OpenAI Chat completion
import os
from openai import AsyncOpenAI  # importing openai for API usage
import chainlit as cl  # importing chainlit for our app
#from chainlit.prompt import Prompt, PromptMessage  # importing prompt tools - Chainlit prompt not needed?
from chainlit.playground.providers import ChatOpenAI  # importing ChatOpenAI tools
from dotenv import load_dotenv

import tiktoken
enc = tiktoken.encoding_for_model("gpt-4-turbo")


load_dotenv()

# Create Object to Access ChatGPT 4 turbo
from langchain_openai import ChatOpenAI
openai_chat_model = ChatOpenAI(model="gpt-4-turbo")

# Load Meta's 10-K PDF
from langchain.document_loaders import PyMuPDFLoader
docs = PyMuPDFLoader("https://d18rn0p25nwr6d.cloudfront.net/CIK-0001326801/c7318154-f6ae-4866-89fa-f0c589f2ee3d.pdf").load()

# Split doc into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4-turbo").encode(
        text,
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
    length_function = tiktoken_len,
)
split_chunks = text_splitter.split_documents(docs)

# Embed each chunk using text embedding 3-small
from langchain_openai.embeddings import OpenAIEmbeddings
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


'''
# Calculate Cosine Similarity Distance Function - not sure if needed if using Qdrant Retriever?
import numpy as np
from numpy.linalg import norm
def cosine_similarity(vec_1, vec_2):
  return np.dot(vec_1, vec_2) / (norm(vec_1) * norm(vec_2))

# Create Retriever - not sure if needed if using Qdrant Retriever?
def retrieve_context(query, embeddings_dict, embedding_model):
  query_vector = embedding_model.embed_query(query)
  max_similarity = -float('inf')
  closest_chunk = ""

  for chunk, chunk_vector in embeddings_dict.items():
    cosine_similarity_score = cosine_similarity(chunk_vector, query_vector)

    if cosine_similarity_score > max_similarity:
      closest_chunk = chunk
      max_similarity = cosine_similarity_score

  return closest_chunk

def simple_rag(query, embeddings_dict, embedding_model, chat_chain):
  context = retrieve_context(query, embeddings_dict, embedding_model)

  response = chat_chain.invoke({"query" : query, "context" : context})

  return_package = {
      "query" : query,
      "response" : response,
      "retriever_context" : context
  }

  return return_package

simple_rag("QUESTION", embeddings_dict, embedding_model, chat_chain)

'''

# Embed chunks into Vector Store
from langchain_community.vectorstores import Qdrant

qdrant_vectorstore = Qdrant.from_documents(
    split_chunks,
    embedding_model,
    location=":memory:",
    collection_name="Meta 2023 10-K",
)

# Setup Retriever
qdrant_retriever = qdrant_vectorstore.as_retriever()
from langchain.retrievers.multi_query import MultiQueryRetriever

llm = openai_chat_model
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=qdrant_retriever, llm=llm
)




# Setup Prompt Template
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = """
CONTEXT:
{context}

QUERY:
{question}

Use the provided context to answer the user's query.

You may not answer the user's query unless there is specific context in the following text.

If you do not know the answer, or cannot answer, please respond with "I don't know".
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

retrieval_augmented_qa_chain = (
    # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
    # "question" : populated by getting the value of the "question" key
    # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
    {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
    # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
    #              by getting the value of the "context" key from the previous step
    | RunnablePassthrough.assign(context=itemgetter("context"))
    # "response" : the "context" and "question" values are used to format our prompt object and then piped
    #              into the LLM and stored in a key called "response"
    # "context"  : populated by getting the value of the "context" key from the previous step
    | {"response": rag_prompt | openai_chat_model, "context": itemgetter("context")}
)


response = retrieval_augmented_qa_chain.invoke({"question" : "Who are Meta's 'Directors' (i.e., members of the Board of Directors)?"})
print(response["response"].content)

response = retrieval_augmented_qa_chain.invoke({"question" : "What was the total value of 'Cash and cash equivalents' as of December 31, 2023?"})
print(response["response"].content)