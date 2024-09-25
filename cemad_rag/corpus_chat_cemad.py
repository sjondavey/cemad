from enum import Enum
from regulations_rag.corpus_chat import CorpusChat
from regulations_rag.rerank import RerankAlgos
import copy

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
                 openai_client, 
                 embedding_parameters,
                 chat_parameters,
                 corpus_index,
                 rerank_algo = RerankAlgos.NONE,   
                 user_name_for_logging = 'test_user'): 
        super().__init__(openai_client, embedding_parameters, chat_parameters, corpus_index, rerank_algo, user_name_for_logging)

    class Prefix(Enum):
        ANSWER =        "ANSWER:"       # from CorpusChat
        SECTION =       "SECTION:"      # from CorpusChat
        NONE =          "NONE:"         # from CorpusChat
        ERROR =         "ERROR:"        # from CorpusChat
        ALTERNATIVE =   "ALTERNATIVE:"  # new path defined in this class



    def select_path_and_execute(self, user_content, df_definitions, df_search_sections, result):
        if not result["success"]:
            logger.error("corpus_chat.resource_augmented_query did not return result[\"success\"] == True.")
            return self.execute_path_for_unsuccessful_rag(user_content, df_definitions, df_search_sections)
            
        if result["path"] == CorpusChat.Prefix.ANSWER.value:
            #   result = {"success": True, "path": "ANSWER:"", "answer": llm_text, "reference": references_as_integers}
            logger.log(DEV_LEVEL, "corpus_chat.resource_augmented_query answered the question using the Retrieved text")
            return self.execute_path_for_successful_rag(user_content, df_definitions, df_search_sections, result)

        elif result["path"] == CorpusChat.Prefix.NONE.value:
            #   result {"success": True, "path": "NONE:"}
            logger.info("corpus_chat.resource_augmented_query was not not able to find relevant information in the retrieved text")
            return self.execute_path_for_no_relevant_information_in_retrieved_text(user_content, df_definitions, df_search_sections)

        elif result["path"] == CorpusChat.Prefix.SECTION.value:
            #   result = {"success": True, "path": "SECTION:", "extract", extract_num_as_int "document": document_name, "section": section_reference} NB the document may not be the same as the document in extract_num_as_int
            logger.log(DEV_LEVEL, f"System requested for more info: Extract {result['extract']} requested section {result['section']}")
            return self.execute_path_for_additional_sections_requested(user_content, df_definitions, df_search_sections, result)

        else:
            logger.error("Note: RAG returned an unexpected response")
            self.append_content("assistant", CorpusChat.Errors.NOT_FOLLOWING_INSTRUCTIONS.value)
            self.system_state = CorpusChat.State.STUCK # We are at a dead end.
            return

    



    def _check_response(self, llm_response_text, df_definitions, df_sections):
        # A valid response will now include the ability to suggest alternative questions
        #{"success": True, "path": "ALTERNATIVE:", "alternatives": pipe_delimited_list_of_alternative_questions} 


        # For reference, here are the return values from CorpusChat._check_response(...)
        #{"success": False, "path": "SECTION:"/"ANSWER:", "llm_followup_instruction": llm_instruction} 
        #{"success": True, "path": "SECTION:", "document": 'GDPR', "section": section_reference}
        #{"success": True, "path": "ANSWER:"", "answer": llm_text, "reference": references_as_integers}
        #{"success": True, "path": "NONE:"}
        if llm_response_text.startswith(CorpusChatCEMAD.Prefix.ALTERNATIVE.value):
            # We have constructed the assistant response manually so we don't need to contend with free text
            prefix = CorpusChatCEMAD.Prefix.ALTERNATIVE.value
            initial_response = llm_response_text[len(prefix):].strip()
            # We know that there are alternatives otherwise we would not be in this path
            pipe_delimited_list_of_alternative_questions = [substr.strip() for substr in initial_response.split('|') if substr]

            return {"success": True, "path": prefix, "alternatives": pipe_delimited_list_of_alternative_questions} 

        else:
            return super()._check_response(llm_response_text, df_definitions, df_sections)

    def _reformat_assistant_answer(self, result, df_definitions, df_search_sections):
        if result["path"] == CorpusChatCEMAD.Prefix.ALTERNATIVE.value:
            # We know that there are alternatives otherwise we would not be in this path
            alternative_questions_with_search_results = result["alternatives"]

            if len(alternative_questions_with_search_results) == 1:
                assistant_content = f"The question you posed did not contain any hits in the database. There are many reasons why this could be the case. Here however is a different phrasing of the question which should find some reference material in the {self.index.corpus_description}. Perhaps try:\n\n" 
                assistant_content = assistant_content + "\n" + alternative_questions_with_search_results[0]
            else: 
                assistant_content = f"The question you posed did not contain any hits in the database. There are many reasons why this could be the case. Here however are different phrasings of the question which should find some reference material in the {self.index.corpus_description}: Perhaps try\n\n" 
                counter = 1
                for question in alternative_questions_with_search_results:
                    assistant_content = assistant_content + "\n" + str(counter) + ") " + question
                    counter = counter + 1

            return assistant_content
        else:
            return super()._reformat_assistant_answer(result, df_definitions, df_search_sections)


    # def select_path_and_execute(self, user_content, df_definitions, df_search_sections, result):
    #     if result["path"] == self.Prefix.ALTERNATIVE.value:
    #         # {"success": True, "path": "ALTERNATIVE:", "alternatives": pipe_delimited_list_of_alternative_questions} 
    #         logger.log(DEV_LEVEL, "CorpusChatCEMAD.resource_augmented_query was not able to find any matches in the database")
    #         return self.execute_path_for_alternative_suggestions(user_content, df_definitions, df_search_sections, result)
            
    #     return super().select_path_and_execute(user_content, df_definitions, df_search_sections, result)

    # def execute_path_for_alternative_suggestions(user_content, df_definitions, df_search_sections, result):
    #     logger.log(DEV_LEVEL, "Executing default CorpusChatCEMAD.execute_path_for_alternative_suggestions() i.e. using the LLM to make alternative versions of the question")
    #     self.system_state = CorpusChat.State.RAG         
    #     self.append_content("user", user_content)       
    #     self.append_content("assistant", CorpusChat.Errors.NO_DATA.value)
    #     return

    """ 
    Override this method if you want to execute something specific when the user question does not result in any hits in the database
    and there is no conversation history that may otherwise allow the system to infer some context and phrase a better question.

    The default behaviour here is to do nothing
    """ 
    def execute_path_no_retrieval_no_conversation_history(self, user_content):
        logger.log(DEV_LEVEL, "Executing CorpusChatCEMAD.execute_path_no_retrieval_no_conversation_history() i.e. Trying to help the user to construct a good question")

        system_content = f"You are assisting a user to construct question that can be used to query a vector database for relevant facts. You will be provided with the original user question which contained no matches from the database that were closer than a cutoff threshold. For context, the vector database contains sections of the {self.index.corpus_description}. \nYour initial task is to determine if the user question is about Exchange Control or not. It is possible the user may be engaging in pleasantries or small talk or may just be testing the bounds of the system. For now please respond with one of only two responses: Relevant if the question, with the conversation history is about Exchange Control or the movement of money in or out of a country or group of countries; or Not Relevant if the topic of the question is anything else. Only respond with Relevant or Not Relevant. Do not add any other text, punctuation or markup to your response."

        # Create a complete list of messages excluding the system message
        messages = self.format_messages_for_openai()
        messages.append({'role': 'user', 'content': user_content})
        
        # Truncate messages list to meet a specific token limit and ensure there is space for the system message
        system_message={'role': 'system', 'content': system_content}
        truncated_messages = self._truncate_message_list([system_message], messages, token_limit=self.token_limit_when_truncating_message_queue)
        # NOTE, the truncated_messages will now contain the system message

        initial_response = self._get_api_response(truncated_messages)

        if initial_response.lower() == 'relevant':
            return self.suggest_alternative_questions(user_content)
        elif initial_response.lower() == 'not relevant':
            return self.hardcode_response_for_question_not_relating_to_excon(user_content)
        else: # instead of placing the system in "stuck" mode, just continue as if the question was not relevant
            return self.hardcode_response_for_question_not_relating_to_excon(user_content)
            # logger.log(DEV_LEVEL, "CorpusChatCEMAD.execute_path_no_retrieval_no_conversation_history() was asked to respond in one of two ways but it ignore this instruction")
            # return self.place_in_stuck_state()


    def suggest_alternative_questions(self, user_content):

        system_content = f"You are assisting a user to construct question that can be used to query a vector database for relevant facts. You will be provided with the original user question which contained no matches from the database that were closer than a cutoff threshold. For context, the vector database contains sections of the {self.index.corpus_description}. \nYour task is to help the user to construct one or more versions of the question which can be answered from the database. Here are some general principles which you could use to help create alternative versions of the question which do have answers in the vector database. \n\
1) CEMAD does not generally refer to countries (other than South Africa) by name. If the user question contains a specific country name other than South Africa, please convert this into  'foreign county' or 'a member of the Common Monetary Area (CMA)'. For example 'Can I open a non-resident rand account for an individual from Eswatini?' should be changed to 'Can I open a non-resident rand account for an individual from the Common Monetary Area?'. \n\
2) CEMAD does not generally refer to specific currencies other than the Rand. Rather it refers to foreign currency or a currency used in the CMA. If you see a reference to a specific currency or currency code that is not Rand, please convert it to 'foreign currency' or 'CMA country currency' as the case may be. For example, 'Can I receive dividends in dollars?' should be changed to 'Can I receive dividends in foreign currency?' or 'Can I receive dividends in a CMA country currency?' because dollars could be US dollars or Namibian dollars. \n\
3) Often the question is not specific about the direction of the currency flow. Inflows and outflows are treated in different sections of CEMAD and if the direction of flow is ambiguous, it may not create a good similarity score with sections in the vector database. Try to add a direction of flow to the question. For example 'Who can trade gold?' should be changed to 'Who can import gold?' and 'Who can export gold?' \n\
4) There are very different exchange control regulations or thresholds for individuals or companies. If the question does not make it clear who the subject of the query is, please add the necessary clarification. For example, 'How much money can I invest offshore?' should be changed to 'How much money can an individual invest offshore?' or 'How much money can a company invest offshore?' depending on the chat context. If there is no context, suggest both options \n\
5) Sometimes the user asks an incomplete question which only makes sense in the context of the entire chat. In this case, try to use the chat history to make a complete question. So a question like 'What is the BOP code for this?' would need to be changed to 'What is the BOP Code for (insert subject here based on the conversation history)?' \n\n\
Please review the user question and provide one or more alternatives to this which are more likely to return a match from CEMAD. Return these in a pipe delimited list with no other text or explanation."

        # Create a complete list of messages excluding the system message
        messages = self.format_messages_for_openai()
        messages.append({'role': 'user', 'content': user_content})
        
        # Truncate messages list to meet a specific token limit and ensure there is space for the system message
        system_message={'role': 'system', 'content': system_content}
        truncated_messages = self._truncate_message_list([system_message], messages, token_limit=self.token_limit_when_truncating_message_queue)
        # NOTE, the truncated_messages will now contain the system message

        initial_response = self._get_api_response(truncated_messages)

        list_of_alternative_questions = []
        if not initial_response:
            list_of_alternative_questions = []
        else:
            list_of_alternative_questions = [substr.strip() for substr in initial_response.split('|') if substr]

        alternative_questions_with_search_results = []
        for question in list_of_alternative_questions:
            workflow_triggered, relevant_definitions, relevant_sections = self.similarity_search(question)
            if len(relevant_definitions) + len(relevant_sections) > 0:
                alternative_questions_with_search_results.append(question)

        if len(alternative_questions_with_search_results) == 0:
            assistant_content = self.Errors.NO_DATA.value
        else:
            assistant_content = self.Prefix.ALTERNATIVE.value + '|'.join(alternative_questions_with_search_results)

        self.system_state = CorpusChat.State.RAG         
        self.append_content("user", user_content)       
        self.append_content("assistant", assistant_content)
        return

    def hardcode_response_for_question_not_relating_to_excon(self, user_content):
        logger.log(DEV_LEVEL, "The user context did not appear to be a question relating to Exchange Control")
        self.system_state = CorpusChat.State.RAG         
        self.append_content("user", user_content)       
        assistant_content = f"ERROR: I am a bot designed to answer questions about the {self.index.corpus_description}. If you ask me a question about that, I will do my best to respond, with a reference. If I cannot find a relevant reference in the document, I have been coded not to respond to the question rather than offing my opinion. Please read the document page for some suggestions if you find this feature frustrating (If you are using this on a mobile phone, look for the little '>' at the top left of your screen)"
        self.append_content("assistant", assistant_content)
        return


    """ 
    Override this method if you want to execute something specific when the user question does not result in any hits in the database
    BUT there IS conversation history that can be used to phrase a better question

    The default behaviour here is to do nothing
    """ 
    def execute_path_no_retrieval_with_conversation_history(self, user_content):
        return self.execute_path_no_retrieval_no_conversation_history(user_content)
        # logger.log(DEV_LEVEL, "Executing default corpus_chat.execute_path_no_retrieval_with_conversation_history() i.e. bypassing the LLM and forcing the assistant to respond with CorpusChat.Errors.NO_DATA.value")
        # self.system_state = CorpusChat.State.RAG         
        # self.append_content("user", user_content)       
        # self.append_content("assistant", CorpusChat.Errors.NO_DATA.value)
        # return
    
    
    """ 
    returns workflow_triggered, df_definitions, df_search_sections
    """
    def execute_workflow(self, workflow_triggered, user_content):
        if workflow_triggered == "documentation":
            #raise NotImplementedError()
            user_content = self.enrich_user_request_for_documentation(user_content)
            workflow_triggered, df_definitions, df_search_sections = self.similarity_search(user_content)
            return workflow_triggered, df_definitions, df_search_sections

        raise NotImplementedError()

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
        messages = self.format_messages_for_openai()
        messages.append({'role': 'user', 'content': user_content})
        
        # Truncate messages list to meet a specific token limit and ensure there is space for the system message
        system_message={'role': 'system', 'content': system_content}
        truncated_messages = self._truncate_message_list([system_message], messages, token_limit=self.token_limit_when_truncating_message_queue)
        # NOTE, the truncated_messages will now contain the system message

        # Generate the enhanced documentation request using the specified AI model
        response = self.openai_client.chat.completions.create(
            model=self.chat_parameters.model,
            temperature=self.chat_parameters.temperature,
            max_tokens=self.chat_parameters.max_tokens,
            messages=truncated_messages
        )

        # Extract the initial response and log information for review
        initial_response = response.choices[0].message.content
        logger.info(f"{self.user_name} original question: {user_content}")
        logger.info(f"System enhanced question: {initial_response}")

        # Check if the response starts as expected; log a warning if not
        if not initial_response.startswith('What documentation is required as evidence for'):
            logger.warning("The function did not enrich the user request for documentation as expected, which may create problems.")

        return initial_response