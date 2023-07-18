import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.prompts import PromptTemplate


def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_tet_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def get_prompt_template():
    input_variables = [],
    template = """ Summarize form of answer """
    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format()
    return formatted_prompt


def main():

    load_dotenv()

    st.set_page_config(page_title="Personal Bot", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversion = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Personal Bot")
    user_question = st.text_input("Asks questions..")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Resume")
        pdf_doc = st.file_uploader("Upload Your Resume")
        if st.button("Sumbit"):
            with st.spinner("processing..."):
                # get pfdf Text
                raw_text = get_pdf_text(pdf_doc)
                text_chunks = get_tet_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vector_store)


if __name__ == "__main__":
    main()
