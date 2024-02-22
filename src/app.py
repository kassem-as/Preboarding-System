import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from dotenv import load_dotenv


if "start" not in st.session_state:
    st.session_state.start = 1
    os.environ["LANGCHAIN_TRACING_V2"] = 'true'
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = "ls__5503692e663d4acaac764ccf25cce896"
    os.environ["LANGCHAIN_PROJECT"] = "preboarding"
    load_dotenv()
    embeddings = CohereEmbeddings(model="embed-multilingual-v3.0")

def get_vectorstore_from_documents():
    loader = TextLoader("documents\Modulhandbuch.txt", encoding='utf-8')
    document = loader.load()

    # transform document to text
    text_content = document[0].page_content

    # split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 700, chunk_overlap = 100)
    text_chunks = text_splitter.split_text(text_content)

    # create a vectorstore from the chunks
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("vector-store")
    return vector_store
        
def get_history_aware_retriever():
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature = 0)

    compressor = CohereRerank()

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=st.session_state.vector_store.as_retriever(search_kwargs={"k":6})
    )

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a text query emphasizing the last message to look up in the vector store. ONLY return the query")
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
        "chat_messages": st.session_state.chat_history[-3:],
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })

    return response['answer']

# app config
st.set_page_config(page_title="Computer Science Preboarding System")
st.title("Computer Science Preboarding System")


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




