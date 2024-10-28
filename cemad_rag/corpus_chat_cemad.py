from enum import Enum
from regulations_rag.corpus_chat import CorpusChat
from regulations_rag.rerank import RerankAlgos, rerank
from regulations_rag.embeddings import get_ada_embedding
from regulations_rag.data_classes import NoAnswerResponse, NoAnswerClassification
from regulations_rag.corpus_chat_tools import CorpusChatData
from regulations_rag.path_search import similarity_search
from cemad_rag.path_suggest_alternatives import suggest_alternative_questions

import logging
logger = logging.getLogger(__name__)
# Create custom log levels for the really detailed logs
DEV_LEVEL = 15
ANALYSIS_LEVEL = 25
logging.addLevelName(DEV_LEVEL, 'DEV')       
logging.addLevelName(ANALYSIS_LEVEL, 'ANALYSIS')       

"""  
I only need this class to implement the workflow hook
"""
class CorpusChatCEMAD(CorpusChat):
    def __init__(self, 
                 embedding_parameters,
                 chat_parameters,
                 corpus_index,
                 rerank_algo = RerankAlgos.NONE,   
                 user_name_for_logging = 'test_user'): 
        super().__init__(embedding_parameters, chat_parameters, corpus_index, rerank_algo, user_name_for_logging)




    """ 
    Override parent method to include suggesting alternative questions when the user question does not result in any hits in the database
    """ 
    def execute_path_no_retrieval_no_conversation_history(self, user_content):
        if self.system_state != CorpusChat.State.RAG:
            return

        super().execute_path_no_retrieval_no_conversation_history(user_content)
        if isinstance(self.messages_intermediate[-1]["assistant_response"], NoAnswerResponse):
            logger.log(ANALYSIS_LEVEL, f"{self.user_name}: query_no_rag_data() responded with a NoAnswerResponse. Evaluating alternative questions...")
            if self.progress_callback:
                self.progress_callback("Unable to answer the question. Evaluating alternative questions...")

            # Remove the last message from the intermediate messages
            del self.messages_intermediate[-1]
            corpus_chat_data = CorpusChatData(corpus_index = self.index, 
                                                    chat_parameters = self.chat_parameters, 
                                                    message_history = self.messages_intermediate, 
                                                    current_user_message = {"role": "user", "content": user_content})
            result = suggest_alternative_questions(corpus_chat_data = corpus_chat_data, 
                                                embedding_parameters = self.embedding_parameters, 
                                                rerank_algo = self.rerank_algo)
            self.append_content(result)
            logger.log(ANALYSIS_LEVEL, f"{self.user_name}: CorpusChatCEMAD.execute_path_no_retrieval_no_conversation_history() replaced the CorpusChat version's answer")

        return

    
    """ 
    returns workflow_triggered, df_definitions, df_search_sections
    """
    def execute_path_workflow(self, workflow_triggered, user_content):
        if workflow_triggered == "documentation":
            logger.log(ANALYSIS_LEVEL, f"{self.user_name}: Enriching user request for documentation ...")
            if self.progress_callback:
                self.progress_callback("Enriching user request for documentation...")

            user_content = self.enrich_user_request_for_documentation(user_content)
            chat_data_for_search = CorpusChatData(corpus_index = self.index, 
                                                    chat_parameters = self.chat_parameters, 
                                                    message_history = self.messages_intermediate, 
                                                    current_user_message = {"role": "user", "content": user_content})
            workflow_triggered, df_definitions, df_search_sections = similarity_search(chat_data_for_search, self.embedding_parameters, self.rerank_algo)

            logger.log(ANALYSIS_LEVEL, f"{self.user_name}: Running RAG for documentation ...")
            if self.progress_callback:
                self.progress_callback("Running RAG for documentation request ...")
            self.run_base_rag_path(user_content, df_definitions, df_search_sections)
            return 

        return super().execute_path_workflow(workflow_triggered, user_content)

    def enrich_user_request_for_documentation(self, user_content):
        """
        Enhances a user's request for documentation based on the conversation history. It constructs a standalone request
        for documentation, utilizing the most recent conversation history to formulate a question that specifies what documentation
        is required.

        Parameters:
        - user_content (str): The latest user content to be used for generating documentation requests.
        - messages_without_rag (list): A list of message dictionaries that exclude RAG content, to be used as conversation history.
        - model_to_use (str, optional): Specifies the AI model to use for generating the documentation request. Defaults to "gpt-3.5-turbo".

        Returns:
        - str: The enhanced documentation request generated by the model.
        """
        logger.info("Enriching user request for documentation based on conversation history.")

        # Preparing the initial system message to guide the model in request generation
        system_content = "You are assisting a user to construct a stand alone request for documentation from a conversation. \
At the end of the conversation they have asked a question about the documentation they require. Your job is to review the conversation history and to respond with this question \
'What documentation is required as evidence for ...' where you need to replace the ellipses with a short description of the most recent conversation history. Try to keep the question short and general."
        
        # Create a complete list of messages excluding the system message
        messages = self.messages_intermediate.copy()
        messages.append({'role': 'user', 'content': user_content})
        
        # Truncate messages list to meet a specific token limit and ensure there is space for the system message
        system_message={'role': 'system', 'content': system_content}
        # NOTE, the truncated_messages will now contain the system message

        # Generate the enhanced documentation request using the specified AI model
        response = self.chat_parameters.get_api_response(system_message = system_message, message_list = messages)

        # Extract the initial response and log information for review
        initial_response = response
        logger.info(f"{self.user_name} original question: {user_content}")
        logger.info(f"System enhanced question: {initial_response}")

        # Check if the response starts as expected; log a warning if not
        if not initial_response.startswith('What documentation is required as evidence for'):
            logger.warning("The function did not enrich the user request for documentation as expected, which may create problems.")

        return initial_response