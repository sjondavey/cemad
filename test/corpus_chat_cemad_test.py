import os
import pandas as pd
from openai import OpenAI
import pytest
from unittest.mock import patch

from regulations_rag.rerank import RerankAlgos

from regulations_rag.corpus_chat import ChatParameters
from regulations_rag.embeddings import  EmbeddingParameters
from regulations_rag.data_classes import AnswerWithRAGResponse, AnswerWithoutRAGResponse, AlternativeQuestionResponse, NoAnswerResponse, NoAnswerClassification

from cemad_rag.cemad_corpus_index import CEMADCorpusIndex
from cemad_rag.corpus_chat_cemad import CorpusChatCEMAD

class TestCEMADChat:
    include_calls_to_api = True

    api_key=os.environ.get("OPENAI_API_KEY")

    chat_parameters = ChatParameters(chat_model = "gpt-4o",  
                                     api_key=api_key, 
                                     temperature = 0, 
                                     max_tokens = 500, 
                                     token_limit_when_truncating_message_queue = 3500)

    embedding_parameters = EmbeddingParameters("text-embedding-3-large", 1024)


    key = os.getenv('excon_encryption_key')
    corpus_index = CEMADCorpusIndex(key)

    rerank_algo  = RerankAlgos.NONE
    if rerank_algo == RerankAlgos.LLM:
        rerank_algo.params["openai_client"] = chat_parameters.openai_client
        rerank_algo.params["model_to_use"] = chat_parameters.model
        rerank_algo.params["user_type"] = corpus_index.user_type
        rerank_algo.params["corpus_description"] = corpus_index.corpus_description
        rerank_algo.params["final_token_cap"] = 5000 # can go large with the new models

    chat = CorpusChatCEMAD(
                      embedding_parameters = embedding_parameters, 
                      chat_parameters = chat_parameters, 
                      corpus_index = corpus_index,
                      rerank_algo = rerank_algo,   
                      user_name_for_logging = 'test_user')

    regression_test_enabled = False

    def test_construction(self):
        assert True

    @patch.object(ChatParameters, 'get_api_response')
    def test_user_provides_input(self, mock_get_api_response):
        self.chat.system_state = self.chat.State.RAG
        self.chat.reset_conversation_history()
        user_content = "can a foreigner buy property in south africa?"
        api_response = "Yes, a foreigner can buy property in South Africa. However, if they require local financial assistance for the acquisition, certain conditions apply. Specifically, the 1:1 ratio will apply for financial assistance unless the non-resident is living and working in South Africa, in which case normal lending criteria apply. Additionally, any financial assistance must be secured by an unencumbered Rand deposit or Rand-based asset of equal or higher value, and the facility may not exceed 100% of the Rand value of funds introduced from abroad and invested locally. Facilities for property acquisition cannot be increased based on property revaluation.Reference: 1"
        path = "ANSWER:"
        mock_get_api_response.return_value = path + api_response
        self.chat.user_provides_input(user_content)
        messages = self.chat.messages_intermediate
        assert "assistant_response" in messages[-1]
        assert isinstance(messages[-1]["assistant_response"], AnswerWithRAGResponse)
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == 'Yes, a foreigner can buy property in South Africa. However, if they require local financial assistance for the acquisition, certain conditions apply. Specifically, the 1:1 ratio will apply for financial assistance unless the non-resident is living and working in South Africa, in which case normal lending criteria apply. Additionally, any financial assistance must be secured by an unencumbered Rand deposit or Rand-based asset of equal or higher value, and the facility may not exceed 100% of the Rand value of funds introduced from abroad and invested locally. Facilities for property acquisition cannot be increased based on property revaluation. \n\nReference: \n\nSection I.1(E) from Currency and Exchange Control Manual for Authorised Dealers: \n\nI.1 Local financial assistance to affected persons and non-residents\n\n&nbsp;&nbsp;&nbsp;&nbsp;(E) Non-residents\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(i) Authorised Dealers may grant or authorise local financial assistance facilities to non-residents in respect of bona fide foreign direct investment in South Africa without restrictions, except where the funds are required for financial transactions and/or the acquisition of residential or commercial property in South Africa, the 1:1 ratio will apply.\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(ii) As an exception to (i) above, Authorised Dealers may grant or authorise local financial assistance facilities to non-residents living and working in South Africa in respect of the acquisition of residential property, subject to normal lending criteria.\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(iii) Any facility being made available to a non-resident party must be secured by an unencumbered Rand deposit or Rand based asset of equal or higher value. In addition, any facility accorded to the non-resident in respect of the aforementioned may not cause the borrower to exceed 100 per cent of the Rand value of funds introduced from abroad and invested locally.\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(iv) If facilities are granted for the acquisition of fixed property, such facilities may not be increased at any stage based on a revaluation of the property in question.  \n\n'
        assert messages[-2]["role"] == "user"
        assert messages[-2]["content"] == user_content

        # Test of the "documentation" workflow. Because this involves a call to an LLM to create the user query, it may fail
        followup_content = "What documentation is required?"
        #enhanced_user_content = "Question: What documentation is required?\n\nExtract 1:\nB.5 Personal transfers by foreign nationals and immigrants\n    (A) Foreign nationals\n        (i) Foreign nationals temporarily in South Africa\n            (a) Foreign nationals (except those who are purely on a temporary visit) are required to declare on arrival in writing to an Authorised Dealer:\n                (aa) whether they are in possession of foreign assets.\n            (b) On receipt of such completed declarations and undertakings, Authorised Dealers may permit such foreign nationals to:\n                (aa) conduct their banking on a resident basis;\n                (bb) dispose of or otherwise invest their foreign assets including foreign cash funds held by them, subsequent accruals, as well as foreign income, without interference from the Financial Surveillance Department.\n                (cc) conduct non-resident or foreign currency accounts in the books of an Authorised Dealer; and\n                (dd) transfer abroad funds accumulated during their stay in South Africa provided that:\n                    (1) the foreign nationals can substantiate the source from which they have acquired such funds; and\n                    (2) the value of such funds is reasonable in relation to their income generating activities in South Africa during the period.\n            (c) The completed declarations and undertakings must be retained by Authorised Dealers, after the permanent departure of such individuals, for a period of five years for inspection purposes.\n            (d) The dispensation in (b) above excludes:\n                (aa) single remittance transactions up to R5 000 per transaction where only the applicant's proof of identity has to be viewed and retained by the Authorised Dealer concerned; and\n                (bb) transactions where a business relationship has been established, which are limited to R5 000 per transaction per day within a limit of R25 000 per applicant per calendar month. The Authorised Dealer must complete the relevant customer due diligence requirements by establishing and verifying the identity of the applicant in terms of section 21 of the FIC Act and obtaining the applicant's residential address information for reporting purposes.\n            (e) It should be noted that while the personal banking of foreign nationals temporarily resident in South Africa may be conducted on a resident basis, any interest held by such individuals in local entities (i.e. legal persons) will be deemed as non-resident for the purposes of local financial assistance.\n            (f) Export of personal effects and other assets\n                (aa) Any household and personal effects, including motor vehicles, may be exported under cover of the prescribed SARS Customs Declaration provided that the goods to be exported have been purchased with funds that would have been transferable or the goods have been imported into South Africa. The individual must, in need, be able to substantiate the importation thereof by the production of documentary evidence.\n            (g) Capital transfers by foreign nationals\n                (aa) Authorised Dealers may permit foreign nationals to retransfer abroad capital that has been introduced into South Africa, provided that they can substantiate the original introduction of such funds."
        rag_system_response = "The documentation required for foreign nationals temporarily in South Africa includes:\n\n1. A written declaration on arrival to an Authorised Dealer stating whether they are in possession of foreign assets.\n2. Substantiation of the source from which they have acquired funds accumulated during their stay in South Africa.\n3. Documentary evidence to substantiate the importation of household and personal effects, including motor vehicles, if needed.\n4. Proof of identity for single remittance transactions up to R5,000.\n5. Customer due diligence documentation, including verification of identity and residential address information, for transactions where a business relationship has been established, limited to R5,000 per transaction per day within a limit of R25,000 per applicant per calendar month.\n6. Substantiation of the original introduction of capital that has been introduced into South Africa for capital transfers.  \Reference: 1"

        first_response = "What documentation is required for a foreigner to buy property in South Africa?"
        mock_get_api_response.side_effect = [first_response, "ANSWER: " + rag_system_response]
        self.chat.user_provides_input(followup_content)
        messages = self.chat.messages_intermediate
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "The documentation required for foreign nationals temporarily in South Africa includes:\n\n1. A written declaration on arrival to an Authorised Dealer stating whether they are in possession of foreign assets.\n2. Substantiation of the source from which they have acquired funds accumulated during their stay in South Africa.\n3. Documentary evidence to substantiate the importation of household and personal effects, including motor vehicles, if needed.\n4. Proof of identity for single remittance transactions up to R5,000.\n5. Customer due diligence documentation, including verification of identity and residential address information, for transactions where a business relationship has been established, limited to R5,000 per transaction per day within a limit of R25,000 per applicant per calendar month.\n6. Substantiation of the original introduction of capital that has been introduced into South Africa for capital transfers.  \\ \n\nReference: \n\nSection I.1(E) from Currency and Exchange Control Manual for Authorised Dealers: \n\nI.1 Local financial assistance to affected persons and non-residents\n\n&nbsp;&nbsp;&nbsp;&nbsp;(E) Non-residents\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(i) Authorised Dealers may grant or authorise local financial assistance facilities to non-residents in respect of bona fide foreign direct investment in South Africa without restrictions, except where the funds are required for financial transactions and/or the acquisition of residential or commercial property in South Africa, the 1:1 ratio will apply.\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(ii) As an exception to (i) above, Authorised Dealers may grant or authorise local financial assistance facilities to non-residents living and working in South Africa in respect of the acquisition of residential property, subject to normal lending criteria.\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(iii) Any facility being made available to a non-resident party must be secured by an unencumbered Rand deposit or Rand based asset of equal or higher value. In addition, any facility accorded to the non-resident in respect of the aforementioned may not cause the borrower to exceed 100 per cent of the Rand value of funds introduced from abroad and invested locally.\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(iv) If facilities are granted for the acquisition of fixed property, such facilities may not be increased at any stage based on a revaluation of the property in question.  \n\n"
        assert messages[-2]["role"] == "user"
        assert messages[-2]["content"] == first_response


    @patch.object(ChatParameters, 'get_api_response')
    def test_execute_path_no_retrieval_no_conversation_history(self, mock_get_api_response):
        self.chat.system_state = self.chat.State.RAG
        self.chat.reset_conversation_history()
        user_content = "Can I use my credit card online?"
        # mock a reasonable response from the LLM
        response = "Can an individual use their credit card for online purchases?|Can a company use its credit card for online purchases?"
        response_as_list = ["Can an individual use their credit card for online purchases?", "Can a company use its credit card for online purchases?"]
        mock_get_api_response.return_value = response

        self.chat.execute_path_no_retrieval_no_conversation_history(user_content)

        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], AlternativeQuestionResponse)
        assert self.chat.messages_intermediate[-1]["assistant_response"].alternatives == response_as_list

        #assert self.chat.messages_intermediate[-1]["content"] == self.chat.Prefix.ALTERNATIVE.value + second_response


        response = ""
        response_as_list = []
        mock_get_api_response.return_value = response

        self.chat.execute_path_no_retrieval_no_conversation_history(user_content)

        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], NoAnswerResponse)
        assert self.chat.messages_intermediate[-1]["assistant_response"].classification == NoAnswerClassification.NO_RELEVANT_DATA


    # def test_execute_path_no_retrieval_with_conversation_history(self):
    #     assert False

    def test_regression(self):
        if not self.regression_test_enabled:
            pytest.skip("Skipping regression test")

        #Because these will make calls to the LLM, they may fail for statistical reasons
        self.chat.reset_conversation_history()
        self.chat._reset_execution_path()

        self.chat.strict_rag = False
        # should answer fine
        self.chat.user_provides_input("When importing goods, can I buy enough foreign currency to cover the associated insurance costs?")
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], AnswerWithRAGResponse)
        expected_path = [
            "CorpusChat.user_provides_input",
            "PathSearch.similarity_search",
            "CorpusChat.run_base_rag_path",
            "PathRAG.perform_RAG_path",
            'PathRAG.resource_augmented_query',
            'PathRAG.check_response_RAG',
            'PathRAG.extract_used_references'
        ]
        assert self.chat.execution_path == expected_path


        # followup question to test workflow
        self.chat._reset_execution_path()
        self.chat.user_provides_input("What documentation is required?")
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert isinstance(self.chat.messages_intermediate[-1]["assistant_response"], AnswerWithRAGResponse)
        expected_path = [
            "CorpusChat.user_provides_input",
            "PathSearch.similarity_search",
            'CorpusChatCEMAD.execute_path_workflow',
            'CorpusChatCEMAD.enrich_user_request_for_documentation',
            "CorpusChat.run_base_rag_path",
            "PathRAG.perform_RAG_path",
            'PathRAG.resource_augmented_query',
            'PathRAG.check_response_RAG',
            'PathRAG.extract_used_references'
        ]
        assert self.chat.execution_path == expected_path


        self.chat.strict_rag = True
