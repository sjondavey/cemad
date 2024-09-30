# https://medium.com/@tophamcherie/authenticating-connecting-to-azure-key-vault-or-resources-programmatically-2e1936618789
# https://learn.microsoft.com/en-us/entra/fundamentals/how-to-create-delete-users
# https://discuss.streamlit.io/t/get-active-directory-authentification-data/22105/57 / https://github.com/kevintupper/streamlit-auth-demo
import logging
from logging_config import setup_logging

import streamlit as st
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

from streamlit_common import setup_for_azure, setup_for_streamlit, load_data

DEV_LEVEL = 15
ANALYSIS_LEVEL = 25
logging.addLevelName(DEV_LEVEL, 'DEV')       
logging.addLevelName(ANALYSIS_LEVEL, 'ANALYSIS')       




# https://docs.streamlit.io/develop/api-reference/cli/run
# If you need to pass an argument to your script, run it as follows:
#     streamlit run your_app.py "my list" of arguments
# Within your script, the following statement will be true:
#     sys.argv[0] == "your_app.py"
#     sys.argv[1] == "my list"
#     sys.argv[2] == "of"
#     sys.argv[3] == "arguments"

st.set_page_config(page_title="Excon Answers", page_icon="./publication_icon.jpg", layout="wide")


if 'temp_logging_file_name' not in st.session_state:
    # Call the setup_logging to configure all loggers
    st.session_state['temp_logging_file_name'] = setup_logging(max_bytes=2 * 1024 * 1024, backup_count=1)
    logger = logging.getLogger(__name__)
    logger.setLevel(ANALYSIS_LEVEL)



if "use_environmental_variables" not in st.session_state:
    st.session_state['use_environmental_variables'] = True 
    if st.session_state['use_environmental_variables']:
        load_dotenv()

if 'log_locally' not in st.session_state:
    st.session_state['log_locally'] = False
    container = os.getenv('BLOB_CONTAINER', 'cemadtest01') # set a default in case 'BLOB_CONTAINER' is not set
    st.session_state['blob_container_name'] = container
    st.session_state['blob_store_key'] = os.getenv("CHAT_BLOB_STORE")
    st.session_state['blob_account_url'] = "https://chatlogsaccount.blob.core.windows.net/"

# Start with username because we need it to create the log file
if 'user_id' not in st.session_state:
    now = datetime.now()
    date_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
    filename = date_time_str + "_user_id.log"
    st.session_state['user_id'] = date_time_str
    logger.log(ANALYSIS_LEVEL, f"New session for user {st.session_state['user_id']}")


if 'service_provider' not in st.session_state:
    # can only be one of 'azure' or 'streamlit'
    if len(sys.argv) > 1 and sys.argv[1] == "azure":
        # run in an azure container using Azure credentials and Azure key vault to save API keys
        st.session_state['service_provider'] = 'azure' 
        setup_for_azure(filename)
    else:
        # run in streamlit community cloud using st.secretes for the username credentials and api keys
        st.session_state['service_provider'] = 'streamlit'
        # Parameter True means to include the username and password
        setup_for_streamlit(True)




if 'selected_model' not in st.session_state.keys():
    #st.session_state['model_options'] = ['gpt-4-0125-preview', 'gpt-4', 'gpt-3.5-turbo']
    st.session_state['model_options'] = ['gpt-4o']
    st.session_state['selected_model'] = 'gpt-4o'
    st.session_state['selected_model_previous'] = 'gpt-4o'


if 'chat' not in st.session_state:
    st.session_state['chat'] = load_data(st.session_state['service_provider'])
    st.session_state['chat'].chat_parameters.model = st.session_state['selected_model']

# list of icons here: https://fonts.google.com/icons
ask_question_page = st.Page("streamlit_pages/1_answer.py", title="Ask a question", default=True, icon=":material/forum:")
toc_question_page = st.Page("streamlit_pages/2_Table_of_Content.py", title="Table of content", icon=":material/list:")
bop_page = st.Page("streamlit_pages/3_BOP_Code_Lookup.py", title="BOP Codes", icon=":material/search:")
section_lookup_page = st.Page("streamlit_pages/4_Lookup_Section.py", title="Section Lookup", icon=":material/filter_alt:")
documentation_page = st.Page("streamlit_pages/5_Read_the_Documents.py", title="Read the documents", icon=":material/article:")
pg = st.navigation({"Other things to do": [ask_question_page, toc_question_page, bop_page, section_lookup_page, documentation_page]})
pg.run()

