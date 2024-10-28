# https://medium.com/@tophamcherie/authenticating-connecting-to-azure-key-vault-or-resources-programmatically-2e1936618789
# https://learn.microsoft.com/en-us/entra/fundamentals/how-to-create-delete-users
# https://discuss.streamlit.io/t/get-active-directory-authentification-data/22105/57 / https://github.com/kevintupper/streamlit-auth-demo
import logging

import streamlit as st
import pandas as pd
import json
import uuid
from streamlit_common import write_session_data_to_blob
from footer import footer


from regulations_rag.rerank import RerankAlgos

from regulations_rag.corpus_chat import ChatParameters
from regulations_rag.embeddings import  EmbeddingParameters

from cemad_rag.cemad_corpus_index import CEMADCorpusIndex
from cemad_rag.corpus_chat_cemad import CorpusChatCEMAD

from regulations_rag.data_classes import AnswerWithRAGResponse, AnswerWithoutRAGResponse, ErrorResponse, ErrorClassification, AlternativeQuestionResponse

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


def display_assistant_response(row, message_index):
    col1, col2 = st.columns([0.95, 0.05])
    
    with col1:
        answer = row["content"]
        st.markdown(answer)
        if "references" in row:
            references = row["references"]
            # references is a dataframe with columns = ["document_key", "document_name", "section_reference", "is_definition", "text"]

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


    
    with col2:
        sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
        selected = st.feedback("thumbs", key=f"feedback_{message_index}")
        if selected is not None:
            log_feedback(message_index, sentiment_mapping[selected])

def log_feedback(message_index, feedback):
    write_session_data_to_blob(f"Feedback for message {message_index}: {feedback}")


# Display or clear chat messages
for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
             display_assistant_response(message, index)
        else:
            st.write(message["content"])

def clear_chat_history():
    # logger.debug("Clearing \'messages\'")
    st.session_state['chat'].reset_conversation_history()
    st.session_state['messages'] = [] 
    write_session_data_to_blob('action: Clear history')

with st.sidebar:

    # Create a custom container for the RAG enforcement option
    rag_container = st.container()
    with rag_container:
        # Define the default mode
        default_mode = False  # Set to True if you want Strict RAG by default

        # Initialize session state for strict RAG mode if it doesn't exist
        if 'strict_rag' not in st.session_state:
            st.session_state.strict_rag = default_mode
        
        # Create a horizontal layout for checkbox and label
        strict_rag = st.checkbox(
            label=":blue[**Strict RAG**]",
            help="When checked, LLM only answers if references are found for each stand alone question. When unchecked, LLM can respond without references.",
            value=st.session_state.strict_rag,
            key='strict_rag_checkbox',
        )
        
        st.session_state.strict_rag = strict_rag        
        # Update the CorpusChat instance with the new strict_rag value
        st.session_state['chat'].strict_rag = st.session_state.strict_rag

    st.button('Clear Chat History', on_click=clear_chat_history)
    st.markdown('**Press the "Clear Chat History" button before you change topic**')
    st.markdown(f'Looking for inspiration? The Reserve Bank has a list of <a href="https://www.resbank.co.za/en/home/what-we-do/financial-surveillance/FinSurvFAQ" target="_blank">Frequently Asked Questions</a>. Try asking one of those here.', unsafe_allow_html=True)

def make_call_to_chat(prompt):
    logger.debug(f"Making call with prompt: {prompt}")                            
    progress_placeholder = st.empty()
    def update_progress(status):
        progress_placeholder.text(f"{status}")
    
    st.session_state['chat'].set_progress_callback(update_progress)

    st.session_state['chat'].user_provides_input(prompt)
    raw_response = st.session_state['chat'].messages_intermediate[-1]

    id = str(uuid.uuid4())
    if "assistant_response" not in raw_response:
        error_response = ErrorResponse(classification=ErrorClassification.ERROR)
        content = error_response.get_text_for_streamlit()
    else:
        content = raw_response["assistant_response"].get_text_for_streamlit()

    row_to_add_to_messages = {"role": "assistant", "content": content, "id": id}
    if isinstance(raw_response["assistant_response"], AnswerWithRAGResponse):
        row_to_add_to_messages["references"] = raw_response["assistant_response"].references
    elif isinstance(raw_response["assistant_response"], AlternativeQuestionResponse):
        row_to_add_to_messages["alternative_phrasing"] = raw_response["assistant_response"].alternatives

    st.session_state['messages'].append(row_to_add_to_messages)

    display_assistant_response(row_to_add_to_messages, len(st.session_state['messages']) - 1)
    write_session_data_to_blob ("assistant: " + raw_response["assistant_response"].create_openai_content() + "\n\n")
    logger.debug("Response added the the queue")
    progress_placeholder.empty() 

# User-provided prompt
prompt = st.chat_input(placeholder="Ask your Exchange Control related question here")

if "user_input" in st.session_state:
    prompt = st.session_state["user_input"]
    del st.session_state["user_input"]  # Clear it after using
    
if prompt is not None and prompt != "":
    log_entry = {"role": "user", "content": prompt}
    st.session_state['messages'].append(log_entry)
    write_session_data_to_blob("User: " + prompt)

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        llm_response_formatted_for_logs = ""

        make_call_to_chat(prompt)
        # with st.spinner("Thinking..."):
        #     make_call_to_chat(prompt)

    
footer()
