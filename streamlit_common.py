import logging
import os
from openai import OpenAI
import platform
import bcrypt

import streamlit as st

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient, ContentSettings

from regulations_rag.rerank import RerankAlgos
from regulations_rag.corpus_chat import ChatParameters
from regulations_rag.embeddings import  EmbeddingParameters

from cemad_rag.cemad_corpus_index import CEMADCorpusIndex
from cemad_rag.corpus_chat_cemad import CorpusChatCEMAD

DEV_LEVEL = 15
ANALYSIS_LEVEL = 25
logging.addLevelName(DEV_LEVEL, 'DEV')       
logging.addLevelName(ANALYSIS_LEVEL, 'ANALYSIS')       

logger = logging.getLogger(__name__)
logger.setLevel(ANALYSIS_LEVEL)

# The container will be the same for all files in the session so only connect to it once.
@st.cache_resource
def _get_blog_container():
    if st.session_state['use_environmental_variables']:
        connection_string = f"DefaultEndpointsProtocol=https;AccountName=chatlogsaccount;AccountKey={st.session_state['blob_store_key']};EndpointSuffix=core.windows.net"
        # Create the BlobServiceClient object using the connection string
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    else:
        tmp_credential = DefaultAzureCredential()
        blob_service_client = BlobServiceClient(st.session_state['blob_account_url'], credential=tmp_credential)

    # Get the container client
    container_client = blob_service_client.get_container_client(st.session_state['blob_container_name'])

    # Check if the container exists, and create it if it doesn't
    if not container_client.exists():
        container_client.create_container()

    return container_client


@st.cache_resource
def _setup_blob_storage_for_logging(filename):
    container_client = _get_blog_container()
    logging_blob = container_client.get_blob_client(filename)
    
    blob_exists = logging_blob.exists()
    if not blob_exists:
        with open(st.session_state['temp_logging_file_name'], "rb") as temp_file:
            container_client.upload_blob(name=filename, data=temp_file, content_settings=ContentSettings(content_type='text/plain'))
    # else:
    #     #existing_content = st.session_state['logging_blob'].download_blob().readall().decode('utf-8')
    #     with open(st.session_state['temp_logging_file_name'], "r") as temp_file:
    #         content = temp_file.read()
    #     st.session_state['logging_blob'].upload_blob(data=content, overwrite=True)
    return logging_blob


# summary data for analysis is sent to individual files per session
# https://stackoverflow.com/questions/77600048/azure-function-logging-to-azure-blob-with-python
def _setup_blob_storage_for_data_collecttion(filename):
    container_client = _get_blog_container()

    st.session_state['output_file'] = container_client.get_blob_client(filename)
    # Check if blob exists, if not create an append blob
    try:
        st.session_state['output_file'].get_blob_properties()  # Check if blob exists
    except:
        # Create an empty append blob if it doesn't exist
        st.session_state['output_file'].create_append_blob()


def setup_for_azure(filename):

    if 'service_provider' not in st.session_state:
        st.session_state['service_provider'] = 'azure'

    # bypass keyvault and set up everything from environmental variables
    if st.session_state['use_environmental_variables']:
        if 'openai_api' not in st.session_state:
            secret_name = "OPENAI_API_KEY_CEMAD"
            openai_api_key = os.getenv(secret_name)
            st.session_state['openai_client'] = OpenAI(api_key = openai_api_key)
        if 'corpus_decryption_key' not in st.session_state:
            secret_name = "DECRYPTION_KEY_CEMAD"
            st.session_state['corpus_decryption_key'] = os.getenv(secret_name)


    else: # use key_vault
        if 'key_vault' not in st.session_state:
            # https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python
            # When the app is running in Azure, DefaultAzureCredential automatically detects if a managed identity exists for the App Service and, if so, uses it to access other Azure resources
            st.session_state['credential'] = DefaultAzureCredential() 
            st.session_state['key_vault'] = "https://cemadragkeyvault.vault.azure.net/"

            # Determine if the app is running locally or on Azure. When run locally, DefaultAzureCredential will default to 
            # environmentcredential and will pull the values AZURE_CLIENT_ID, AZURE_TENANT_ID and AZURE_CLIENT_SECRET from the
            # .env file 
            if os.getenv('AZURE_ENVIRONMENT') == 'local':
                st.session_state['app_path'] = "http://localhost:8501"

            else: # folder in Azure
                st.session_state['app_path'] = "https://cemadrag-c8cve3anewdpcdhf.southafricanorth-01.azurewebsites.net"

            if 'corpus_decryption_key' not in st.session_state:
                secret_client = SecretClient(vault_url=st.session_state['key_vault'], credential=st.session_state['credential'])
                st.session_state['corpus_decryption_key'] = secret_client.get_secret(secret_name).value

        if 'openai_api' not in st.session_state:
            secret_client = SecretClient(vault_url=st.session_state['key_vault'], credential=st.session_state['credential'])
            api_key = secret_client.get_secret(secret_name)
            st.session_state['openai_client'] = OpenAI(api_key = api_key.value)

    _setup_blob_storage_for_data_collecttion(filename)
    st.session_state['app_log_blob_file_name'] = "app_log_data.txt"
    st.session_state['logging_blob'] = _setup_blob_storage_for_logging(st.session_state['app_log_blob_file_name'])

    if not "password_correct" in st.session_state: # No passwords yet in Azure but passwords required for other pages
        st.session_state["password_correct"] = True



def setup_for_streamlit(insist_on_password = False):
    if 'service_provider' not in st.session_state:
        st.session_state['service_provider'] = 'streamlit'

    # test to see if we are running locally or on the streamlit cloud
    if 'app_path' not in st.session_state:
        test_variable = platform.processor()
        if test_variable: # running locally
            st.session_state['app_path'] = "http://localhost:8501"
        else: # we are on the cloud
            st.session_state['app_path'] = "https://exconmanualchat.streamlit.app/"

    if 'output_folder' not in st.session_state:
        st.session_state['output_folder'] = "./user_data/"

    if 'corpus_decryption_key' not in st.session_state:
        st.session_state['corpus_decryption_key'] = st.secrets["index"]["decryption_key"]

    if 'openai_api' not in st.session_state:
        st.session_state['openai_client'] = OpenAI(api_key = st.secrets['openai']['OPENAI_API_KEY'])

        if not insist_on_password:
            if "password_correct" not in st.session_state.keys():
                st.session_state["password_correct"] = True
        else:
            ## Password
            def check_password():
                """Returns `True` if the user had a correct password."""

                def login_form():
                    """Form with widgets to collect user information"""
                    with st.form("Credentials"):
                        st.text_input("Username", key="username")
                        st.text_input("Password", type="password", key="password")
                        st.form_submit_button("Log in", on_click=password_entered)

                def password_entered():
                    """Checks whether a password entered by the user is correct."""
                    pwd_raw = st.session_state['password']
                    if st.session_state["username"] in st.secrets[
                        "passwords"
                    ] and bcrypt.checkpw(
                        pwd_raw.encode(),
                        st.secrets.passwords[st.session_state["username"]].encode(),
                    ):
                        st.session_state["password_correct"] = True
                        logger.log(ANALYSIS_LEVEL, f"New questions From: {st.session_state['username']}")
                        del st.session_state["password"]  # Don't store the username or password.
                        del pwd_raw
                        st.session_state["user_id"] = st.session_state["username"] 
                        del st.session_state["username"]
                        
                    else:
                        st.session_state["password_correct"] = False

                # Return True if the username + password is validated.
                if st.session_state.get("password_correct", False):
                    return True

                # Show inputs for username + password.
                login_form()
                if "password_correct" in st.session_state:
                    st.error("😕 User not known or password incorrect")
                return False

            if not check_password():
                st.stop()

@st.cache_resource
def load_cemad_corpus_index(key):
    logger.log(ANALYSIS_LEVEL, f"*** Loading cemad corpis index. This should only happen once")
    return CEMADCorpusIndex(key)

def load_data(service_provider):
    with st.spinner(text="Loading the excon documents and index - hang tight! This should take 5 seconds."):
        corpus_index = load_cemad_corpus_index(st.session_state['corpus_decryption_key'])

        rerank_algo = RerankAlgos.LLM
        rerank_algo.params["openai_client"] = st.session_state['openai_client']
        rerank_algo.params["model_to_use"] = st.session_state['selected_model']
        rerank_algo.params["user_type"] = corpus_index.user_type
        rerank_algo.params["corpus_description"] = corpus_index.corpus_description
        rerank_algo.params["final_token_cap"] = 5000 # can go large with the new models

        embedding_parameters = EmbeddingParameters("text-embedding-3-large", 1024)
        chat_parameters = ChatParameters(chat_model = "gpt-4o", temperature = 0, max_tokens = 500)
        
        chat = CorpusChatCEMAD(openai_client = st.session_state['openai_client'],
                          embedding_parameters = embedding_parameters, 
                          chat_parameters = chat_parameters, 
                          corpus_index = corpus_index,
                          rerank_algo = rerank_algo,   
                          user_name_for_logging=st.session_state["user_id"])

        return chat


def write_data_to_output(text):
    if st.session_state['service_provider'] == 'azure':
        # bespoke data per user
        st.session_state['output_file'].append_block(text + "\n")
        # logs

        blob_exists = st.session_state['logging_blob'].exists()
        if not blob_exists:
            with open(st.session_state['temp_logging_file_name'], "rb") as temp_file:
                container_client = _get_blog_container()
                container_client.upload_blob(name=st.session_state['app_log_blob_file_name'], data=temp_file, content_settings=ContentSettings(content_type='text/plain'))
        else:
            with open(st.session_state['temp_logging_file_name'], "r") as temp_file:
                content = temp_file.read()
            st.session_state['logging_blob'].upload_blob(data=content, overwrite=True)
