
import streamlit as st
import traceback
import os
import sys
from datetime import datetime

from streamlit_common import setup_for_azure, \
                             setup_for_streamlit, \
                             load_data, \
                             setup_log_storage, \
                             write_global_data_to_blob, \
                             write_session_data_to_blob

import logging
from logging_config import setup_logging
DEV_LEVEL = 15
ANALYSIS_LEVEL = 25
logging.addLevelName(DEV_LEVEL, 'DEV')       
logging.addLevelName(ANALYSIS_LEVEL, 'ANALYSIS')       
# Call the setup_logging function to configure ALL loggers

try:
    # https://docs.streamlit.io/develop/api-reference/cli/run
    # If you need to pass an argument to your script, run it as follows:
    #     streamlit run your_app.py "my list" of arguments
    # Within your script, the following statement will be true:
    #     sys.argv[0] == "your_app.py"
    #     sys.argv[1] == "my list"
    #     sys.argv[2] == "of"
    #     sys.argv[3] == "arguments"

    st.set_page_config(page_title="Excon Answers", page_icon="./publication_icon.jpg", layout="wide")

    # I need the logging file in the session state so this is rerun every session. setup_logging is 
    # also a cache resource so it will just keep returning the same file name 
    if 'global_logging_file_name' not in st.session_state: 
        st.session_state['global_logging_file_name'] = setup_logging(max_bytes=2 * 1024 * 1024, backup_count=1)
        # Add a logger for this file
        logger = logging.getLogger(__name__)
        logger.setLevel(ANALYSIS_LEVEL)
        logger.log(ANALYSIS_LEVEL, f"logging file name: {st.session_state['global_logging_file_name']}")
        

    # Start with username because we need it to create the log file
    if 'user_id' not in st.session_state:
        now = datetime.now()
        date_time_str = now.strftime("%Y_%m_%d_%H_%M_%S")
        st.session_state['user_id'] = date_time_str
        st.session_state['blob_name_for_session_logs'] = date_time_str + "_user_id.log"
        logger.log(ANALYSIS_LEVEL, f"New session for user {st.session_state['user_id']}")
        
    if 'service_provider' not in st.session_state:
        # can only be one of 'azure' or 'streamlit'
        if len(sys.argv) > 1 and sys.argv[1] == "azure":
            # run in an azure container using Azure credentials and Azure key vault to save API keys
            st.session_state['service_provider'] = 'azure' 
            setup_for_azure()
        else:
            # TODO: Streamlit stuff all needs to be checked
            # run in streamlit community cloud using st.secretes for the username credentials and api keys
            st.session_state['service_provider'] = 'streamlit'
            # Parameter True means to include the username and password
            setup_for_streamlit(True)

    if 'chat' not in st.session_state:
        st.session_state['chat'] = load_data()

    # list of icons here: https://fonts.google.com/icons
    ask_question_page = st.Page("streamlit_pages/1_answer.py", title="Ask a question", default=True, icon=":material/forum:")
    toc_question_page = st.Page("streamlit_pages/2_Table_of_Content.py", title="Table of content", icon=":material/list:")
    bop_page = st.Page("streamlit_pages/3_BOP_Code_Lookup.py", title="BOP Codes", icon=":material/search:")
    section_lookup_page = st.Page("streamlit_pages/4_Lookup_Section.py", title="Section Lookup", icon=":material/filter_alt:")
    documentation_page = st.Page("streamlit_pages/5_Read_the_Documents.py", title="Read the documents", icon=":material/article:")
    pg = st.navigation({"Other things to do": [ask_question_page, toc_question_page, bop_page, section_lookup_page, documentation_page]})
    pg.run()

    setup_log_storage(st.session_state['blob_name_for_session_logs'])
    # copy the existing local log file to blob storage. This is done at the beginning of the session becasue
    # streamlit does not offer a direct session termination even. It is not idea as the logs will always be
    # one session behind
    if "global_logs_copied_to_storage" not in st.session_state:
        write_global_data_to_blob()
        st.session_state["global_logs_copied_to_storage"] = True

except Exception as e:
    error_traceback = traceback.format_exc()
    error_string = "user: " + st.session_state["user_id"] + ": \n" + error_traceback
    logger = logging.getLogger(__name__)
    logger.error(error_string)
    write_global_data_to_blob()
    write_session_data_to_blob(error_string)
    raise e