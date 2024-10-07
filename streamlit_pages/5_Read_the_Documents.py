import os
import streamlit as st
from streamlit_common import _get_blob_for_session_data_logging

st.title('Documentation')

# Currently only set up for azure using environmental variables. Other options need to be built
def setup_log_storage():
    if st.session_state['service_provider'] == 'azure':
        if st.session_state['use_environmental_variables'] == True:
            if 'blob_account_url' not in st.session_state:
                st.session_state['blob_account_url'] = "https://chatlogsaccount.blob.core.windows.net/"
                st.session_state['blob_container_name'] = os.getenv('BLOB_CONTAINER', 'cemadtest01') # set a default in case 'BLOB_CONTAINER' is not set
                st.session_state['blob_store_key'] = os.getenv("CHAT_BLOB_STORE")
                #st.session_state['blob_client_for_session_data'] = _get_blob_for_session_data_logging(filename)
                #st.session_state['blob_name_for_global_logs'] = "app_log_data.txt"
                #st.session_state['blob_client_for_global_data'] = _get_blob_for_global_logging(st.session_state['blob_name_for_global_logs'])


def upload_logs_to_blob_and_cleanup():
    # Directory holding session logs
    log_directory = '/tmp/session_logs'
    if os.path.exists(log_directory):
        for log_file in os.listdir(log_directory):
            if log_file != os.path.basename(st.session_state['local_session_data']): # don't copy the current users local data
                # check it is not this logfile 
                file_path = os.path.join(log_directory, log_file)
                if os.path.isfile(file_path):
                    # Get blob client and upload the file only when button is pressed
                    blob_client = _get_blob_for_session_data_logging(log_file)
                    with open(file_path, 'rb') as data:
                        blob_client.upload_blob(data, blob_type="AppendBlob", overwrite=True)
                        #blob_client.upload_blob(data, overwrite=True) 
                    if log_file != "app_log_data.txt":
                        os.remove(file_path)  # Clean up the local file after upload


with st.sidebar:
    st.markdown('Thanks for reading the instructions. You are one of a small minority of people and you deserve a gold star!')



d = '''This Question Answering service is an example of **Retrieval Augmented Generation (RAG)**. It uses a Large Language Model to answer questions based on its reference material (which you can see the in the Table of Contents page). This service is not official nor endorsed by anyone relevant. Its answers should be treated as guidance, not law. If you use these answers as the basis to perform an action and that action is illegal, there is nobody to sue or join with you in your court case. You will be on your own, with only your blind faith in Large Language Models for company. 

To reduce the chance of incorrect answers, a key feature of this service is its ability not to answer when it cannot find relevant source material. There may be times when this feature feels more like a bug. In those cases, there are a few things you can try:

- **Specify the direction of currency flow**: Inflows and outflows are treated in different sections of CEMAD. If the question is ambiguous about the direction of the flow, the system may not retrieve the relevant documentation. For example, instead of asking "Who can trade gold?" try asking "Who can import gold?" or "Who can export gold?"

- **Ensure the question is complete**: If the question only makes sense in the context of the chat (e.g., "What is the BOP code for this?"), rephrase it as a complete question. For example, ask "What is the BOP code for gold imports?"

- **Avoid specific country names**: CEMAD typically does not refer to countries other than South Africa by name. If your question includes a specific country name, change it to "foreign country" or "a member of the Common Monetary Area (CMA)." For example, "Can I open a non-resident rand account for an individual from Eswatini?" should be changed to "Can I open a non-resident rand account for an individual from the Common Monetary Area?"

- **Avoid specific currency names**: CEMAD generally doesn't reference specific currencies other than the Rand. Instead, it uses terms like "foreign currency" or "CMA country currency." For example, "Can I receive dividends in dollars?" should be changed to "Can I receive dividends in foreign currency?" or "Can I receive dividends in a CMA country currency?" since "dollars" could refer to US dollars or Namibian dollars.

- **Clarify the subject of the query**: There are different exchange control regulations and thresholds for individuals and companies. If the question doesn't make this distinction clear, add the necessary context. For example, "How much money can I invest offshore?" should be clarified as "How much money can an individual invest offshore?" or "How much money can a company invest offshore?"

If you want to get some insight into how this app was built, have a look [here](https://www.aleph-one.co)
'''

st.markdown(d)

# Sidebar button that triggers the upload and cleanup process
if st.sidebar.button("Persist Data"):
    setup_log_storage()
    upload_logs_to_blob_and_cleanup()
