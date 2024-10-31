import logging
from regulations_rag.data_classes import AlternativeQuestionResponse, NoAnswerResponse, NoAnswerClassification
from regulations_rag.embeddings import get_ada_embedding, EmbeddingParameters
from regulations_rag.rerank import RerankAlgos

from regulations_rag.corpus_chat_tools import ChatParameters
from cemad_rag.cemad_corpus_index import CEMADCorpusIndex


logger = logging.getLogger(__name__)
DEV_LEVEL = 15
ANALYSIS_LEVEL = 25
logging.addLevelName(DEV_LEVEL, 'DEV')       
logging.addLevelName(ANALYSIS_LEVEL, 'ANALYSIS')       


class PathSuggestAlternatives:

    def __init__(self, chat_parameters: ChatParameters, corpus_index: CEMADCorpusIndex, embedding_parameters: EmbeddingParameters, rerank_algo: RerankAlgos):
        self.chat_parameters = chat_parameters
        self.corpus_index = corpus_index
        self.embedding_parameters = embedding_parameters
        self.rerank_algo = rerank_algo

    def remove_rag_data_from_message_history(self, message_history: list):
        stripped_message_history = []
        for message in message_history:
            stripped_message = {"role": message["role"], "content": message["content"]}
            stripped_message_history.append(stripped_message)
        return stripped_message_history

    def suggest_alternative_questions(self, message_history: list, current_user_message: dict):
        logger.log(ANALYSIS_LEVEL, "Suggesting alternative questions")

        system_content = f"""You are an AI assistant helping to construct alternative questions for querying a vector database containing sections of the {self.corpus_index.corpus_description}. Review the conversation history and the user's latest question, then suggest one or more alternative versions that are more likely to match the database content. Consider the following:

    1. Contextualize incomplete questions using the conversation history (e.g., "What about children?" becomes "What is the travel allowance for children under 18 years old?" depending on the chat history).
    2. Replace specific country names (except South Africa) with "foreign country" or "Common Monetary Area (CMA) member".
    3. Convert specific foreign currencies to "foreign currency" or "CMA country currency".
    4. Clarify the direction of currency flow (inflow/outflow) if ambiguous.
    5. Specify whether the subject is an individual or a company if unclear.
    6. Ensure questions are complete and self-contained.

    Provide alternative questions in a pipe-delimited list without additional text or explanations."""

        
        # Truncate messages list to meet a specific token limit and ensure there is space for the system message
        system_message=[{'role': 'system', 'content': system_content}]

        messages = self.remove_rag_data_from_message_history(message_history)
        messages.append(current_user_message)
        
        initial_response = self.chat_parameters.get_api_response(system_message, messages)

        list_of_alternative_questions = []
        if not initial_response:
            list_of_alternative_questions = []
        else:
            list_of_alternative_questions = [substr.strip() for substr in initial_response.split('|') if substr]

        alternative_questions_with_search_results = []
        for question in list_of_alternative_questions:
            question_embedding = get_ada_embedding(self.chat_parameters.openai_client, question, self.embedding_parameters.model, self.embedding_parameters.dimensions)        
            relevant_definitions = self.corpus_index.get_relevant_definitions(user_content = question, user_content_embedding = question_embedding, threshold = self.embedding_parameters.threshold_definitions)
            relevant_sections = self.corpus_index.get_relevant_sections(user_content = question, 
                                                                user_content_embedding = question_embedding, 
                                                                threshold = self.embedding_parameters.threshold, 
                                                                rerank_algo = self.rerank_algo)            
            number_of_hits = len(relevant_definitions) + len(relevant_sections) 
            if number_of_hits > 0:
                alternative_questions_with_search_results.append([question, number_of_hits])

        # Sort alternative_questions_with_search_results by descending number_of_hits
        alternative_questions_with_search_results.sort(key=lambda x: x[1], reverse=True)

        # Create a list of only the questions in the same order
        sorted_questions = [item[0] for item in alternative_questions_with_search_results]
        
        if len(alternative_questions_with_search_results) == 0:
            assistant_response = NoAnswerResponse(classification = NoAnswerClassification.NO_RELEVANT_DATA)
            response = {"role": "assistant", "content": assistant_response.create_openai_content(), "assistant_response": assistant_response}
        else:
            assistant_response = AlternativeQuestionResponse(alternatives = sorted_questions)
            response = {"role": "assistant", "content": assistant_response.create_openai_content(), "assistant_response": assistant_response}

        return response


