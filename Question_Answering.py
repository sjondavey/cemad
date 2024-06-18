import logging

import streamlit as st
import openai
import os
import bcrypt
from openai import OpenAI

DEV_LEVEL = 15
ANALYSIS_LEVEL = 25
logging.addLevelName(DEV_LEVEL, 'DEV')       
logging.addLevelName(ANALYSIS_LEVEL, 'ANALYSIS')       

logging.basicConfig(level=ANALYSIS_LEVEL)
logger = logging.getLogger(__name__)
logger.setLevel(ANALYSIS_LEVEL)

from regulations_rag.rerank import RerankAlgos

from cemad_rag.cemad_chat import CEMADChat


if 'user_id' not in st.session_state:
    st.session_state['user_id'] = ""


### Password
def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        pwd_raw = st.session_state['password']
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and bcrypt.checkpw(
            pwd_raw.encode(),
            st.secrets.passwords[st.session_state["username"]].encode(),
        ):
            st.session_state["password_correct"] = True
            logger.log(ANALYSIS_LEVEL, f"New questions From: {st.session_state['username']}")
            del st.session_state["password"]  # Don't store the username or password.
            del pwd_raw
            st.session_state["user_id"] = st.session_state["username"] 
            del st.session_state["username"]
            
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False

if not check_password():
    st.stop()


# App title - Must be first Streamlit command
st.set_page_config(page_title="💬 Excon Manual Question Answering")

st.title('Dealer Manual: Question Answering')

if 'openai_api' not in st.session_state:
    st.session_state['openai_client'] = OpenAI(api_key = st.secrets['openai']['OPENAI_API_KEY'])

def load_data():
    logger.debug(f'--> cache_resource called again to reload data')
    with st.spinner(text="Loading the excon documents and index - hang tight! This should take 5 seconds."):

        rerank_algo = RerankAlgos.LLM
        excon = CEMADChat(openai_client = st.session_state['openai_client'],
                          decryption_key = st.secrets['index']['decryption_key'],
                          rerank_algo = rerank_algo,
                          user_name_for_logging=st.session_state["user_id"])


        return excon

if 'excon' not in st.session_state:
    logger.debug('Adding \'Excon\' to keys')
    st.session_state['excon'] = load_data()


if 'selected_model' not in st.session_state.keys():
    st.session_state['model_options'] = ['gpt-4-0125-preview', 'gpt-4', 'gpt-3.5-turbo']
    st.session_state['selected_model'] = 'gpt-4-0125-preview'
    st.session_state['selected_model_previous'] = 'gpt-4-0125-preview'
    st.session_state['excon'].chat_parameters.model = st.session_state['selected_model']



st.write(f"I am a bot designed to answer questions based on the {st.session_state['excon'].index.regulation_name}. How can I assist today?")
# Credentials
with st.sidebar:

    #st.subheader('Models and parameters')
        
    st.session_state['selected_model'] = st.sidebar.selectbox('Choose a model', st.session_state['model_options'], key='user_selected_model')
    if st.session_state['selected_model'] != st.session_state['selected_model_previous']:
        st.session_state['selected_model_previous'] = st.session_state['selected_model']
        st.session_state['excon'].chat_parameters.model = st.session_state['selected_model']
        logger.log(ANALYSIS_LEVEL, f"{st.session_state['user_id']} changed model and is now using {st.session_state['selected_model']}")


    temperature = 0.0
    max_length = 500 # Note. If you increase this, you need to amend the two instances of the following lines of code in chat_bot.py
        #   if (model_to_use == "gpt-3.5-turbo" or model_to_use == "gpt-4") and total_tokens > 3500 and model_to_use!="gpt-3.5-turbo-16k":
        #                     logger.warning("!!! NOTE !!! You have a very long prompt. Switching to the gpt-3.5-turbo-16k model")
        #                     model_to_use = "gpt-3.5-turbo-16k"    
        # Because the 'hard coded' number 3500 plus this max_lenght gets very close to the default model's token limit
    # max_length = st.sidebar.slider('max_length', min_value=32, max_value=2048, value=512, step=8)
    # temperature = st.sidebar.slider('temperature', min_value=0.00, max_value=2.0, value=0.0, step=0.01)
    st.divider()
        
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    logger.debug("Adding \'messages\' to keys")
    st.session_state['excon'].reset_conversation_history()
    st.session_state['messages'] = [] 

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    logger.debug("Clearing \'messages\'")
    st.session_state['excon'].reset_conversation_history()
    st.session_state['messages'] = [] 
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# User-provided prompt
if prompt := st.chat_input():
    logger.debug(f"st.chat_input() called. Value returned is: {prompt}")        
    if prompt is not None and prompt != "":
        st.session_state['messages'].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            placeholder = st.empty()

            with st.spinner("Thinking..."):
                logger.debug(f"Making call to excon manual with prompt: {prompt}")
                response = st.session_state['excon'].chat_completion(user_content = prompt)
                logger.debug(f"Response received")
                logger.debug(f"Text Returned from excon manual chat: {response}")
                placeholder.markdown(response)
            st.session_state['messages'].append({"role": "assistant", "content": response})
            logger.debug("Response added the the queue")
    
