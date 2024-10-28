import streamlit as st
from regulations_rag.embeddings import get_ada_embedding, get_closest_nodes
import pandas as pd
import json
from streamlit_common import write_session_data_to_blob

st.title('Search BOP Codes')


@st.cache_resource(show_spinner=False)
def load_bop_codes_data():
    with st.spinner(text="Loading the BOP codes - hang tight! This should take a few seconds."):
        path_to_bop_codes_as_parquet_file = "./inputs/index/bopcodes.parquet"
        df = pd.read_parquet(path_to_bop_codes_as_parquet_file, engine="pyarrow")
        return df

if 'bop_codes' not in st.session_state:
    st.session_state['bop_codes'] = load_bop_codes_data()


# Store LLM generated responses
if "bop_lookup" not in st.session_state.keys():
    st.session_state['bop_lookup'] = [{"role": "assistant", "content": "Which code are you looking for?"}]

for message in st.session_state['bop_lookup']:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state['bop_lookup'] = [{"role": "assistant", "content": "Which code are you looking for?"}]
    write_session_data_to_blob('action_bop_lookup: Clear history')

with st.sidebar:
    st.button('Clear Lookup History', on_click=clear_chat_history)
    st.markdown(f'Type the topic you want to search for in the message bar below, and you will see a list of codes that are similar to the text you entered. You can search using a simple word, like "gold" or "VAT", or by typing a full sentence.')
    st.markdown('Please note: There is no chat functionality on this page—only lookup functionality.')


# User-provided prompt
if prompt := st.chat_input(disabled= ('password_correct' not in st.session_state or not st.session_state["password_correct"])): 
    st.session_state['bop_lookup'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)        
        write_session_data_to_blob(f"user_bop_lookup: {prompt}")


# Generate a new response if last message is not from assistant
if st.session_state['bop_lookup'][-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            openai_client = st.session_state['chat'].openai_client 
            model=st.session_state['chat'].embedding_parameters.model
            dimensions=st.session_state['chat'].embedding_parameters.dimensions
            threshold = st.session_state['chat'].embedding_parameters.threshold
            question_embedding = get_ada_embedding(openai_client=openai_client, text=prompt, model=model, dimensions=dimensions)
            closest_nodes = get_closest_nodes(st.session_state['bop_codes'], "embedding", question_embedding, threshold = 1.0)
            closest_nodes = closest_nodes.nsmallest(16, 'cosine_distance') 

    relevant_columns = ["category", "sub-category", "category description", "inward or outward"]
    df = closest_nodes[relevant_columns]
    st_df = st.dataframe(closest_nodes[relevant_columns], hide_index = True)
    st.session_state['bop_lookup'].append({"role": "assistant", "content": df})




