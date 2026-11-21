import os
from typing import TypedDict, Annotated
from langchain.graph.messages import add_message
from langgraph.graph import StateGraph, START, END
form langchain_groq import ChatGroq
ffrom langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from lanchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()