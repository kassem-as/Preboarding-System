import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from dotenv import load_dotenv


if "start" not in st.session_state:
    st.session_state.start = 1
    load_dotenv()
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

def get_vectorstore_from_documents():
    docs = []
    for doc in [TextLoader("documents\Modulhandbuch.txt", encoding='utf-8'), TextLoader("documents\Prüfungsordnung.txt", encoding='utf-8')]:
        docs.extend(doc.load())
    # texts = [doc.page_content for doc in docs]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 900, chunk_overlap = 300)
    text_chunks = text_splitter.split_documents(docs)
    

    
    # split the document into chunks
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    # text_chunks = text_splitter.split_documents(docs)

    # create a vectorstore from the chunks
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    vector_store.save_local("vector-store")
    return vector_store
    
def get_history_aware_retriever():
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

    compressor = CohereRerank(model="rerank-multilingual-v2.0")

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k":20})
    )

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_messages"),
        ("user", "{input}"),
        ("system", "In the conversation above, the user and the system discuss various topics. When the user asks follow-up questions, they might use pronouns ('it', 'they', 'them') referring back to previously mentioned subjects, or they might ask new, standalone questions that do not require context from earlier in the conversation. Your task is to either: a) replace pronouns with the specific subject or noun previously discussed for clarity, or b) repeat the question as it is if it's a standalone question not requiring modification. Do not answer the question. Focus solely on clarifying the question or repeating it verbatim.")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, compression_retriever, prompt)
    return history_aware_retriever

def get_stuff_documents_chain():
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's question based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return stuff_documents_chain

def get_response(user_query, retrieval_chain):
    response = retrieval_chain.invoke({
        "chat_messages": st.session_state.chat_history,
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })

    return response['answer']

# app config
st.set_page_config(page_title="Computer Science Preboarding System")
st.title("CS Preboarding System")


# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, how can I help you today?"),
    ]

# sidebar
with st.sidebar:
    if st.button("Start new chat"):
        st.session_state.chat_history = [
            AIMessage(content="Hello, how can I help you today?"),
    ]


if "vector_store" not in st.session_state:
    if os.path.exists('vector-store'):
        st.session_state.vector_store = FAISS.load_local("vector-store", embeddings)
    else:
        st.session_state.vector_store = get_vectorstore_from_documents()
    

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# creating the retrieval chain
if "retrieval_chain" not in st.session_state:
        history_aware_retriever = get_history_aware_retriever()
        stuff_documents_chain = get_stuff_documents_chain()
        st.session_state.retrieval_chain = create_retrieval_chain(history_aware_retriever,stuff_documents_chain)


# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    with st.chat_message("Human"):
        st.write(user_query)
    
    response = get_response(user_query, st.session_state.retrieval_chain)

    with st.chat_message("AI"):
        st.write(response)

    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))




