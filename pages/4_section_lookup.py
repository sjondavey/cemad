import streamlit as st
from cemad_rag.corpus_chat import CorpusChat

if 'chat' not in st.session_state:
    st.switch_page('question_answering.py')


st.title('Dealer Manual: Section Lookup')


# Store LLM generated responses
if "messages_lookup" not in st.session_state.keys():
    #excon.reset_conversation_history()
    st.session_state.messages_lookup = [{"role": "assistant", "content": "Which section are you after? Please use the format A.1(A)(i)(a)(aa)"}]

for message in st.session_state.messages_lookup:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    #excon.reset_conversation_history()
    st.session_state.messages_lookup = [{"role": "assistant", "content": "Which section are you after? Please use the format A.1(A)(i)(a)(aa)"}]
st.sidebar.button('Clear Lookup History', on_click=clear_chat_history)


# User-provided prompt
if prompt := st.chat_input(disabled= ('password_correct' not in st.session_state or not st.session_state["password_correct"])): 
    st.session_state.messages_lookup.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

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
                # formatted_response = ''
                # lines = response.split('\n')
                # for line in lines:
                #     spaces = len(line) - len(line.lstrip())
                #     formatted_response += '- ' + '&nbsp;' * spaces + line.lstrip() + "  \n"
            placeholder.markdown(formatted_response)
    st.session_state.messages_lookup.append({"role": "assistant", "content": formatted_response})



