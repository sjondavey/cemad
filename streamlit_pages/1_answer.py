# https://medium.com/@tophamcherie/authenticating-connecting-to-azure-key-vault-or-resources-programmatically-2e1936618789
# https://learn.microsoft.com/en-us/entra/fundamentals/how-to-create-delete-users
# https://discuss.streamlit.io/t/get-active-directory-authentification-data/22105/57 / https://github.com/kevintupper/streamlit-auth-demo
import logging

import streamlit as st
import pandas as pd
import json

from streamlit_common import write_session_data_to_blob
from footer import footer


from regulations_rag.rerank import RerankAlgos

from regulations_rag.corpus_chat import ChatParameters
from regulations_rag.embeddings import  EmbeddingParameters

from cemad_rag.cemad_corpus_index import CEMADCorpusIndex
from cemad_rag.corpus_chat_cemad import CorpusChatCEMAD

logger = logging.getLogger(__name__)

st.title('Ask me a question about South African Exchange Control')
st.markdown(f'A bot that answers questions based on the {st.session_state["chat"].index.corpus_description}. This bot is **not** endorsed by anyone official.')

temperature = 0.0
max_length = 1000 
        
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    logger.debug("Adding \'messages\' to keys")
    #file_handler.error("Something only the filehander should see")
    st.session_state['chat'].reset_conversation_history()
    st.session_state['messages'] = [] 

def create_user_question(prompt):
    st.session_state["user_input"] = prompt

def display_assistant_response(row):
    answer = row["content"]
    references = row.get("section_reference") # This will return the value if the key exists, or None if it doesn't.
    st.markdown(answer)
    if references is not None and not references.empty:
        for index, row in references.iterrows():
            document_name = row["document_name"]
            document_key = row["document_key"]
            section_reference = row["section_reference"]
            is_definition = row["is_definition"]
            if is_definition: # just get the unformatted version because get_text returns all the definitions
                text = row['text']
            else:
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
    elif "alternative_phrasing" in row:
        for alternative in row["alternative_phrasing"]:
            with st.spinner("Thinking..."):
                st.button(alternative, on_click=create_user_question, args=(alternative, ))

# Display or clear chat messages
# https://discuss.streamlit.io/t/chat-message-assistant-component-getting-pushed-into-user-message/57231
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
             display_assistant_response(message)
        else:
            st.write(message["content"])

def clear_chat_history():
    # logger.debug("Clearing \'messages\'")
    st.session_state['chat'].reset_conversation_history()
    st.session_state['messages'] = [] 
    write_session_data_to_blob('{"role": "action", "content": "Clear history"}')

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
with st.sidebar:
    st.markdown('Answers *only* if references can be found.')
    st.markdown('**Press the "Clear Chat History" button before you change topic**')
    st.markdown(f'Looking for inspiration? The Reserve Bank has a list of <a href="https://www.resbank.co.za/en/home/what-we-do/financial-surveillance/FinSurvFAQ" target="_blank">Frequently Asked Questions</a>. Try asking one of those here.', unsafe_allow_html=True)

def make_call_to_chat(prompt):
    logger.debug(f"Making call with prompt: {prompt}")                            
    st.session_state['chat'].user_provides_input(prompt)
    raw_response = st.session_state['chat'].messages_intermediate[-1]
    llm_reply = raw_response['content']
    df_definitions = raw_response['definitions']
    df_search_sections = raw_response['sections']
    response_dict = st.session_state['chat']._check_response(llm_reply, df_definitions, df_search_sections)
    row_to_add_to_messages = {}
    llm_response_formatted_for_logs = st.session_state['chat']._reformat_assistant_answer(response_dict, df_definitions, df_search_sections)
    if response_dict['path'] == st.session_state['chat'].Prefix.ALTERNATIVE.value:
        llm_answer, other_suggestions = st.session_state['chat']._extract_assistant_answer_and_references(response_dict, df_definitions, df_search_sections)
        assistant_response = llm_answer # llm_response_formatted_for_logs
        row_to_add_to_messages = {"role": "assistant", "content": assistant_response, "section_reference": pd.DataFrame(), "alternative_phrasing": other_suggestions}
    else:
        llm_answer, df_references_list = st.session_state['chat']._extract_assistant_answer_and_references(response_dict, df_definitions, df_search_sections)
        row_to_add_to_messages = {"role": "assistant", "content": response_dict['answer'], "section_reference": df_references_list}

    st.session_state['messages'].append(row_to_add_to_messages)

    display_assistant_response(row_to_add_to_messages)
    log_entry = {"role": "assistant", "content": llm_response_formatted_for_logs}
    write_session_data_to_blob(json.dumps(log_entry))
    logger.debug("Response added the the queue")


# User-provided prompt
prompt = st.chat_input(placeholder="Ask your Exchange Control related question here")

if "user_input" in st.session_state:
    prompt = st.session_state["user_input"]
    del st.session_state["user_input"]  # Clear it after using
    
if prompt is not None and prompt != "":
    log_entry = {"role": "user", "content": prompt}
    st.session_state['messages'].append(log_entry)
    write_session_data_to_blob(json.dumps(log_entry))

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        llm_response_formatted_for_logs = ""

        with st.spinner("Thinking..."):
            make_call_to_chat(prompt)

    
footer()