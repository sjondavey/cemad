import json
import streamlit as st
from cemad_rag.corpus_chat_cemad import CorpusChatCEMAD

from streamlit_common import write_data_to_output


st.title('Dealer Manual: Section Lookup')


# Store LLM generated responses
if "messages_lookup" not in st.session_state.keys():
    st.session_state.messages_lookup = [{"role": "assistant", "content": "Which section are you after?"}]

for message in st.session_state.messages_lookup:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages_lookup = [{"role": "assistant", "content": "Which section are you after?"}]
    write_data_to_output('{"role": "action_section_lookup", "content": "Clear history"}')


with st.sidebar:
    st.button('Clear Lookup History', on_click=clear_chat_history)
    st.markdown('The Currency and Exchange Control Manual uses references like: "A.1(A)(i)(a)(aa)". It begins with an uppercase letter followed by a full stop. After this, a number may follow, though it is not always used (e.g., Section C has no number, and is referenced simply as C.(A) or C.(G)). After the letter or number, all subsequent index values are enclosed in round brackets.')
    st.markdown('You only need to include as many index levels as required.')


# User-provided prompt
if prompt := st.chat_input(disabled= ('password_correct' not in st.session_state or not st.session_state["password_correct"])): 
    st.session_state.messages_lookup.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
        log_entry = {"role": "user_section_lookup", "content": prompt}
        write_data_to_output(json.dumps(log_entry))


# Generate a new response if last message is not from assistant
if st.session_state.messages_lookup[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            placeholder = st.empty()
            prompt = st.session_state['chat'].corpus.get_document("CEMAD").reference_checker.extract_valid_reference(prompt)
            if not prompt:
                formatted_response = "I was not able to extract a valid index from the value you input. Please try using the format A.1(A)(i)(a)(aa)."
            else:
                response = st.session_state['chat'].corpus.get_text("CEMAD", prompt)
                formatted_response = response
            placeholder.markdown(formatted_response)
    st.session_state.messages_lookup.append({"role": "assistant", "content": formatted_response})



