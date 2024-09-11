# https://medium.com/@tophamcherie/authenticating-connecting-to-azure-key-vault-or-resources-programmatically-2e1936618789
# https://learn.microsoft.com/en-us/entra/fundamentals/how-to-create-delete-users
# https://discuss.streamlit.io/t/get-active-directory-authentification-data/22105/57 / https://github.com/kevintupper/streamlit-auth-demo
import logging

import streamlit as st
import os
import re


from streamlit_common import setup_for_azure, setup_for_streamlit, load_data

from regulations_rag.rerank import RerankAlgos

from regulations_rag.corpus_chat import ChatParameters
from regulations_rag.embeddings import  EmbeddingParameters

from cemad_rag.cemad_corpus_index import CEMADCorpusIndex
from cemad_rag.corpus_chat_cemad import CorpusChatCEMAD


DEV_LEVEL = 15
ANALYSIS_LEVEL = 25
logging.addLevelName(DEV_LEVEL, 'DEV')       
logging.addLevelName(ANALYSIS_LEVEL, 'ANALYSIS')       

logging.basicConfig(level=ANALYSIS_LEVEL)
logger = logging.getLogger(__name__)
logger.setLevel(ANALYSIS_LEVEL)


# App title - Must be first Streamlit command
st.set_page_config(page_title="💬 Excon Manual Question Answering", layout="wide")

if 'service_provider' not in st.session_state:
    # can only be one of 'azure' or 'streamlit'
    st.session_state['service_provider'] = 'azure'
    #st.session_state['service_provider'] = 'streamlit'

    if st.session_state['service_provider'] == 'azure':
        setup_for_azure()
    elif st.session_state['service_provider'] == 'streamlit':
        setup_for_streamlit(True)
    else:
        st.markdown("An internal error occurred. The service_provider variable can only be 'azure' or 'streamlit'")
        st.stop()

if 'user_id' not in st.session_state:
    st.session_state['user_id'] = "TODO: Get Name"


if 'selected_model' not in st.session_state.keys():
    #st.session_state['model_options'] = ['gpt-4-0125-preview', 'gpt-4', 'gpt-3.5-turbo']
    st.session_state['model_options'] = ['gpt-4o']
    st.session_state['selected_model'] = 'gpt-4o'
    st.session_state['selected_model_previous'] = 'gpt-4o'


if 'chat' not in st.session_state:
    st.session_state['chat'] = load_data(st.session_state['service_provider'])
    st.session_state['chat'].chat_parameters.model = st.session_state['selected_model']

# used if I need hyperlinks from text: https://discuss.streamlit.io/t/hyperlink-to-another-streamlit-page-inside-a-text/65463/8
if 'page_names' not in st.session_state: 
    st.session_state['page_table_of_content'] = 'Table_of_Content'
    st.session_state['page_bop_lookup'] = 'BOP_Code_Lookup'
    st.session_state['page_lookup_section'] = 'Lookup_Section'
    st.session_state['page_read_the_documents'] = 'Read_the_Documents'


st.title('CEMAD: Question Answering')

st.markdown(f'I am a bot designed to answer questions based on {st.session_state["chat"].index.corpus_description}.')
st.markdown('**I can only answer questions if I can find a reference in my source documents**, which you can view on the "Table of Contents" page. If you receive a response indicating an inability to find relevant documentation, please refer to the "Read the Documents" page.')
st.markdown(f'Looking for inspiration? The Reserve Bank has a list of <a href="https://www.resbank.co.za/en/home/what-we-do/financial-surveillance/FinSurvFAQ" target="_blank">Frequently Asked Questions</a>. Try asking one of those here!', unsafe_allow_html=True)
st.markdown('**I work best if you press the "Clear Chat History" button when you want to ask a question about a new topic**')

temperature = 0.0
max_length = 1000 
        
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    logger.debug("Adding \'messages\' to keys")
    st.session_state['chat'].reset_conversation_history()
    st.session_state['messages'] = [] 

def display_assistant_response(row):
    answer = row["content"]
    references = row.get("section_reference") # This will return the value if the key exists, or None if it doesn't.
    st.markdown(answer)
    if references is not None and not references.empty:
        for index, row in references.iterrows():
            document_name = row["document_name"]
            document_key = row["document_key"]
            section_reference = row["section_reference"]
            logger.error(section_reference)
            #text = row["text"]
            text = st.session_state['chat'].index.corpus.get_text(document_key, section_reference, add_markdown_decorators=True, add_headings=True, section_only=False)
            reference_string = ""
            if row["is_definition"]:
                if section_reference == "":
                    reference_string += f"The definitions in {document_name}  \n"
                else:
                    reference_string += f"Definition {section_reference} from {document_name}  \n"
            else:
                if section_reference == "":
                    reference_string += f"The document {document_name}  \n"
                else:
                    reference_string += f"Section {section_reference} from {document_name}  \n"
            with st.expander(reference_string):
                st.markdown(text, unsafe_allow_html=True)


# Display or clear chat messages
# https://discuss.streamlit.io/t/chat-message-assistant-component-getting-pushed-into-user-message/57231
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
             display_assistant_response(message)
        else:
            st.write(message["content"])

def clear_chat_history():
    logger.debug("Clearing \'messages\'")
    st.session_state['chat'].reset_conversation_history()
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

            with st.spinner("Thinking..."):
                logger.debug(f"Making call with prompt: {prompt}")                            
                st.session_state['chat'].user_provides_input(prompt)
                raw_response = st.session_state['chat'].messages_intermediate[-1]
                llm_reply = raw_response['content']
                df_definitions = raw_response['definitions']
                df_search_sections = raw_response['sections']
                response_dict = st.session_state['chat']._check_response(llm_reply, df_definitions, df_search_sections)
                row_to_add_to_messages = {}
                if response_dict['path'] == st.session_state['chat'].Prefix.ALTERNATIVE.value:
                    assistant_response = st.session_state['chat']._reformat_assistant_answer(response_dict, df_definitions, df_search_sections)
                    row_to_add_to_messages = {"role": "assistant", "content": assistant_response, "section_reference": pd.DataFrame()}
                else:
                    llm_answer, df_references_list = st.session_state['chat']._extract_assistant_answer_and_references(response_dict, df_definitions, df_search_sections)
                    row_to_add_to_messages = {"role": "assistant", "content": response_dict['answer'], "section_reference": df_references_list}
                st.session_state['messages'].append(row_to_add_to_messages)
                logger.debug(f"Response received")
                logger.debug(f"Text Returned from GDPR chat: {llm_reply}")
            display_assistant_response(row_to_add_to_messages)
            logger.debug("Response added the the queue")
    
