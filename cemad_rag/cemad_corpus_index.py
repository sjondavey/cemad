import logging
import os
import pandas as pd

from regulations_rag.file_tools import load_parquet_data
from regulations_rag.corpus_index import DataFrameCorpusIndex
from cemad_rag.cemad_corpus import CEMADCorpus

# Create a logger for this module
logger = logging.getLogger(__name__)
DEV_LEVEL = 15
logging.addLevelName(DEV_LEVEL, 'DEV')       

class CEMADCorpusIndex(DataFrameCorpusIndex):
    def __init__(self, key):
        corpus = CEMADCorpus("./cemad_rag/documents/")
        index_folder = "./inputs/index/"
        index_df = pd.DataFrame()
        list_of_index_files = ["ad_index.parquet", "ad_index_plus.parquet"]
        # for filename in os.listdir(index_folder):
        for filename in list_of_index_files:
            if filename.endswith(".parquet"):  
                filepath = os.path.join(index_folder, filename)
                df = load_parquet_data(filepath, key)
                index_df = pd.concat([index_df, df], ignore_index = True)

        df_index_df = pd.DataFrame()
        list_of_definitions_index_files = ["ad_definitions.parquet"]
        for filename in list_of_definitions_index_files:
            if filename.endswith(".parquet"):  
                filepath = os.path.join(index_folder, filename)
                df = pd.read_parquet(filepath, engine="pyarrow") # not encrypted
                df_index_df = pd.concat([df_index_df, df], ignore_index = True)

        user_type = "an Authorised Dealer (AD)" 
        corpus_description = "South African \'Currency and Exchange Manual for Authorised Dealers\' (CEMAD)"

        definitions = df_index_df
        definitions["text"] = definitions["definition"]
        index = index_df
        workflow = pd.read_parquet(os.path.join(index_folder, "workflow.parquet"), engine="pyarrow")

        super().__init__(user_type, corpus_description, corpus, definitions, index, workflow)

#     def get_relevant_definitions(self, user_content, user_content_embedding, threshold):
#     def cap_rag_section_token_length(self, relevant_sections, capped_number_of_tokens):
#     def get_relevant_sections(self, user_content, user_content_embedding, threshold, rerank_algo = RerankAlgos.NONE):
#     def get_relevant_workflow(self, user_content_embedding, threshold):

