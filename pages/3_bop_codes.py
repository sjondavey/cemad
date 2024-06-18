import streamlit as st
from regulations_rag.embeddings import get_ada_embedding, get_closest_nodes
import pandas as pd

st.title('Dealer Manual: BOP Codes')


with st.sidebar:
    st.title('💬 BOP codes lookup')

@st.cache_resource(show_spinner=False)
def load_bop_codes_data():
    with st.spinner(text="Loading the BOP codes - hang tight! This should take a few seconds."):
        path_to_bop_codes_as_parquet_file = "./inputs/bopcodes.parquet"
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
    #excon.reset_conversation_history()
    st.session_state['bop_lookup'] = [{"role": "assistant", "content": "Which code are you looking for?"}]
st.sidebar.button('Clear Lookup History', on_click=clear_chat_history)


# User-provided prompt
if prompt := st.chat_input(disabled= ('password_correct' not in st.session_state or not st.session_state["password_correct"])): 
    st.session_state['bop_lookup'].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state['bop_lookup'][-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            openai_client = st.session_state['excon'].openai_client 
            model=st.session_state['excon'].embedding_parameters.model
            dimensions=st.session_state['excon'].embedding_parameters.dimensions
            threshold = st.session_state['excon'].embedding_parameters.threshold
            question_embedding = get_ada_embedding(openai_client=openai_client, text=prompt, model=model, dimensions=dimensions)
            closest_nodes = get_closest_nodes(st.session_state['bop_codes'], "embedding", question_embedding, threshold = 1.0)
            closest_nodes = closest_nodes.nsmallest(16, 'cosine_distance') 

    relevant_columns = ["category", "sub-category", "category description", "inward or outward"]
    df = closest_nodes[relevant_columns]
    st_df = st.dataframe(closest_nodes[relevant_columns], hide_index = True)
    st.session_state['bop_lookup'].append({"role": "assistant", "content": df})



