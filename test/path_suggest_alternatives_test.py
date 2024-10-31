import os
from unittest.mock import patch
from regulations_rag.corpus_chat_tools import ChatParameters
from regulations_rag.data_classes import AlternativeQuestionResponse, NoAnswerResponse, NoAnswerClassification
from regulations_rag.embeddings import EmbeddingParameters
from regulations_rag.rerank import RerankAlgos
from cemad_rag.path_suggest_alternatives import PathSuggestAlternatives
from cemad_rag.cemad_corpus_index import CEMADCorpusIndex

@patch.object(ChatParameters, 'get_api_response')
def test_suggest_alternative_questions(mock_get_api_response):

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable is not set. Skipping this test.")

    chat_parameters = ChatParameters(chat_model = "gpt-4o-mini",  
                                     api_key=api_key, 
                                     temperature = 0, 
                                     max_tokens = 500, 
                                     token_limit_when_truncating_message_queue = 3500)

    key = os.getenv('excon_encryption_key')
    corpus_index = CEMADCorpusIndex(key)

    rerank_algo  = RerankAlgos.NONE
    # if rerank_algo == RerankAlgos.LLM:
    #     rerank_algo.params["openai_client"] = chat_parameters.openai_client
    #     rerank_algo.params["model_to_use"] = chat_parameters.model
    #     rerank_algo.params["user_type"] = corpus_index.user_type
    #     rerank_algo.params["corpus_description"] = corpus_index.corpus_description
    #     rerank_algo.params["final_token_cap"] = 5000 # can go large with the new models


    user_content = "Can I use my credit card online?"
    user_message = {"role": "user", "content": user_content}
    embedding_parameters = EmbeddingParameters(embedding_model = "text-embedding-3-large", embedding_dimensions = 1024)

    path = PathSuggestAlternatives(chat_parameters = chat_parameters, 
                                                        corpus_index = corpus_index, 
                                                        embedding_parameters = embedding_parameters, 
                                                        rerank_algo = rerank_algo)

    # mock an empty response from the LLM
    mock_get_api_response.return_value = ""
    response = path.suggest_alternative_questions(message_history = [], current_user_message = user_message)

    assert response["role"] == "assistant"
    assert isinstance(response["assistant_response"], NoAnswerResponse)
    assert response["assistant_response"].classification == NoAnswerClassification.NO_RELEVANT_DATA
    assert response["content"] == NoAnswerClassification.NO_RELEVANT_DATA.value

    api_response = "Can an individual use their credit card for online purchases?|Can a company use its credit card for online purchases?"
    api_response_as_list = ["Can an individual use their credit card for online purchases?","Can a company use its credit card for online purchases?"]
    mock_get_api_response.return_value = api_response
    response = path.suggest_alternative_questions(message_history = [], current_user_message = user_message)
    assert response["role"] == "assistant"
    assert isinstance(response["assistant_response"], AlternativeQuestionResponse)
    assert response["assistant_response"].alternatives == api_response_as_list
    assert response["content"] == response["assistant_response"].create_openai_content()
