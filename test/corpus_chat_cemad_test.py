import os
import pandas as pd
from openai import OpenAI
import pytest
from unittest.mock import patch

from regulations_rag.rerank import RerankAlgos

from regulations_rag.corpus_chat import ChatParameters
from regulations_rag.embeddings import  EmbeddingParameters

from cemad_rag.cemad_corpus_index import CEMADCorpusIndex
from cemad_rag.corpus_chat_cemad import CorpusChatCEMAD

class TestCEMADChat:
    include_calls_to_api = True
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
    #chat_parameters = ChatParameters(chat_model = "gpt-3.5-turbo", temperature = 0, max_tokens = 500)
    chat_parameters = ChatParameters(chat_model = "gpt-4o", temperature = 0, max_tokens = 500)
    embedding_parameters = EmbeddingParameters("text-embedding-3-large", 1024)

    key = os.getenv('excon_encryption_key')
    corpus_index = CEMADCorpusIndex(key)

    rerank_algo  = RerankAlgos.LLM
    if rerank_algo == RerankAlgos.LLM:
        rerank_algo.params["openai_client"] = openai_client
        rerank_algo.params["model_to_use"] = chat_parameters.model
        rerank_algo.params["user_type"] = corpus_index.user_type
        rerank_algo.params["corpus_description"] = corpus_index.corpus_description
        rerank_algo.params["final_token_cap"] = 5000 # can go large with the new models

    chat = CorpusChatCEMAD(openai_client = openai_client, 
                      embedding_parameters = embedding_parameters, 
                      chat_parameters = chat_parameters, 
                      corpus_index = corpus_index,
                      rerank_algo = rerank_algo,   
                      user_name_for_logging = 'test_user')


    def test_construction(self):
        assert True

    def test__check_response(self):
        df_definitions = pd.DataFrame() 
        df_sections = pd.DataFrame() 
        prefix = self.chat.Prefix.ALTERNATIVE.value
        alt_questions = " Can an individual use their credit card for online purchases? | Can a company use its credit card for online purchases?"
        alt_questions_as_array = [substr.strip() for substr in alt_questions.split('|') if substr]
        llm_response_text = prefix + alt_questions
        response = self.chat._check_response(llm_response_text, df_definitions, df_sections)
        assert response["success"] == True
        assert response["path"] == prefix
        assert len(response["alternatives"]) == 2
        assert response["alternatives"][0] == alt_questions_as_array[0]
        assert response["alternatives"][1] == alt_questions_as_array[1]

    def test__reformat_assistant_answer(self):
        df_definitions = pd.DataFrame() 
        df_sections = pd.DataFrame() 
        prefix = self.chat.Prefix.ALTERNATIVE.value
        alt_questions = " Can an individual use their credit card for online purchases? | Can a company use its credit card for online purchases?"
        alt_questions_as_array = [substr.strip() for substr in alt_questions.split('|') if substr]
        result = {"success": True, "path": prefix, "alternatives": alt_questions_as_array} 
        text_response_for_openai = self.chat._reformat_assistant_answer(result, df_definitions, df_sections)

        expected_response = f"The question you posed did not contain any hits in the database. There are many reasons why this could be the case. Here however are different phrasings of the question which should find some reference material in the {self.chat.index.corpus_description}" 
        counter = 1
        for question in alt_questions_as_array:
            expected_response = expected_response + "\n" + str(counter) + ") " + question
            counter = counter + 1
        assert text_response_for_openai == expected_response
        # hard coded
        expected_response = "The question you posed did not contain any hits in the database. There are many reasons why this could be the case. Here however are different phrasings of the question which should find some reference material in the South African 'Currency and Exchange Manual for Authorised Dealers' (CEMAD)\n1) Can an individual use their credit card for online purchases?\n2) Can a company use its credit card for online purchases?"
        assert text_response_for_openai == expected_response
        
        # now with only one alternative
        alt_questions = " Can an individual use their credit card for online purchases?"
        alt_questions_as_array = [substr.strip() for substr in alt_questions.split('|') if substr]
        result = {"success": True, "path": prefix, "alternatives": alt_questions_as_array} 
        text_response_for_openai = self.chat._reformat_assistant_answer(result, df_definitions, df_sections)
        # hard coded
        expected_response = "The question you posed did not contain any hits in the database. There are many reasons why this could be the case. Here however is a different phrasing of the question which should find some reference material in the South African 'Currency and Exchange Manual for Authorised Dealers' (CEMAD)\nCan an individual use their credit card for online purchases?"
        assert text_response_for_openai == expected_response
        


    @patch.object(CorpusChatCEMAD, '_get_api_response')
    def test_user_provides_input(self, mock__get_api_response):
        self.chat.system_state = self.chat.State.RAG
        self.chat.reset_conversation_history()
        user_content = "can a foreigner buy property in south africa?"
        enhanced_user_content = "Question: can a foreigner buy property in south africa?\n\nExtract 1:\nI.1 Local financial assistance to affected persons and non-residents\n    (E) Non-residents\n        (i) Authorised Dealers may grant or authorise local financial assistance facilities to non-residents in respect of bona fide foreign direct investment in South Africa without restrictions, except where the funds are required for financial transactions and/or the acquisition of residential or commercial property in South Africa, the 1:1 ratio will apply.\n        (ii) As an exception to (i) above, Authorised Dealers may grant or authorise local financial assistance facilities to non-residents living and working in South Africa in respect of the acquisition of residential property, subject to normal lending criteria.\n        (iii) Any facility being made available to a non-resident party must be secured by an unencumbered Rand deposit or Rand based asset of equal or higher value. In addition, any facility accorded to the non-resident in respect of the aforementioned may not cause the borrower to exceed 100 per cent of the Rand value of funds introduced from abroad and invested locally.\n        (iv) If facilities are granted for the acquisition of fixed property, such facilities may not be increased at any stage based on a revaluation of the property in question."
        api_response = "Yes, a foreigner can buy property in South Africa. However, if they require local financial assistance for the acquisition, certain conditions apply. Specifically, the 1:1 ratio will apply for financial assistance unless the non-resident is living and working in South Africa, in which case normal lending criteria apply. Additionally, any financial assistance must be secured by an unencumbered Rand deposit or Rand-based asset of equal or higher value, and the facility may not exceed 100% of the Rand value of funds introduced from abroad and invested locally. Facilities for property acquisition cannot be increased based on property revaluation.Reference: 1"
        rag_system_response = "Yes, a foreigner can buy property in South Africa. However, if they require local financial assistance for the acquisition, certain conditions apply. Specifically, the 1:1 ratio will apply for financial assistance unless the non-resident is living and working in South Africa, in which case normal lending criteria apply. Additionally, any financial assistance must be secured by an unencumbered Rand deposit or Rand-based asset of equal or higher value, and the facility may not exceed 100% of the Rand value of funds introduced from abroad and invested locally. Facilities for property acquisition cannot be increased based on property revaluation.  \nReference:  \nSection I.1(E) from Currency and Exchange Control Manual for Authorised Dealers"
        path = "ANSWER:"
        mock__get_api_response.return_value = path + api_response
        self.chat.user_provides_input(user_content)
        messages = self.chat.format_messages_for_openai()
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"].strip() == rag_system_response
        assert messages[-2]["role"] == "user"
        assert messages[-2]["content"].strip() == enhanced_user_content

        # Test of the "documentation" workflow. Because this involves a call to an LLM to create the user query, it may fail
        followup_content = "What documentation is required?"
        enhanced_user_content = "Question: What documentation is required?\n\nExtract 1:\nB.5 Personal transfers by foreign nationals and immigrants\n    (A) Foreign nationals\n        (i) Foreign nationals temporarily in South Africa\n            (a) Foreign nationals (except those who are purely on a temporary visit) are required to declare on arrival in writing to an Authorised Dealer:\n                (aa) whether they are in possession of foreign assets.\n            (b) On receipt of such completed declarations and undertakings, Authorised Dealers may permit such foreign nationals to:\n                (aa) conduct their banking on a resident basis;\n                (bb) dispose of or otherwise invest their foreign assets including foreign cash funds held by them, subsequent accruals, as well as foreign income, without interference from the Financial Surveillance Department.\n                (cc) conduct non-resident or foreign currency accounts in the books of an Authorised Dealer; and\n                (dd) transfer abroad funds accumulated during their stay in South Africa provided that:\n                    (1) the foreign nationals can substantiate the source from which they have acquired such funds; and\n                    (2) the value of such funds is reasonable in relation to their income generating activities in South Africa during the period.\n            (c) The completed declarations and undertakings must be retained by Authorised Dealers, after the permanent departure of such individuals, for a period of five years for inspection purposes.\n            (d) The dispensation in (b) above excludes:\n                (aa) single remittance transactions up to R5 000 per transaction where only the applicant's proof of identity has to be viewed and retained by the Authorised Dealer concerned; and\n                (bb) transactions where a business relationship has been established, which are limited to R5 000 per transaction per day within a limit of R25 000 per applicant per calendar month. The Authorised Dealer must complete the relevant customer due diligence requirements by establishing and verifying the identity of the applicant in terms of section 21 of the FIC Act and obtaining the applicant's residential address information for reporting purposes.\n            (e) It should be noted that while the personal banking of foreign nationals temporarily resident in South Africa may be conducted on a resident basis, any interest held by such individuals in local entities (i.e. legal persons) will be deemed as non-resident for the purposes of local financial assistance.\n            (f) Export of personal effects and other assets\n                (aa) Any household and personal effects, including motor vehicles, may be exported under cover of the prescribed SARS Customs Declaration provided that the goods to be exported have been purchased with funds that would have been transferable or the goods have been imported into South Africa. The individual must, in need, be able to substantiate the importation thereof by the production of documentary evidence.\n            (g) Capital transfers by foreign nationals\n                (aa) Authorised Dealers may permit foreign nationals to retransfer abroad capital that has been introduced into South Africa, provided that they can substantiate the original introduction of such funds."
        api_response = "The documentation required for foreign nationals temporarily in South Africa includes:\n\n1. A written declaration on arrival to an Authorised Dealer stating whether they are in possession of foreign assets.\n2. Substantiation of the source from which they have acquired funds accumulated during their stay in South Africa.\n3. Documentary evidence to substantiate the importation of household and personal effects, including motor vehicles, if needed.\n4. Proof of identity for single remittance transactions up to R5,000.\n5. Customer due diligence documentation, including verification of identity and residential address information, for transactions where a business relationship has been established, limited to R5,000 per transaction per day within a limit of R25,000 per applicant per calendar month.\n6. Substantiation of the original introduction of capital that has been introduced into South Africa for capital transfers.  Reference:  2"
        rag_system_response = "The documentation required for foreign nationals temporarily in South Africa includes:\n\n1. A written declaration on arrival to an Authorised Dealer stating whether they are in possession of foreign assets.\n2. Substantiation of the source from which they have acquired funds accumulated during their stay in South Africa.\n3. Documentary evidence to substantiate the importation of household and personal effects, including motor vehicles, if needed.\n4. Proof of identity for single remittance transactions up to R5,000.\n5. Customer due diligence documentation, including verification of identity and residential address information, for transactions where a business relationship has been established, limited to R5,000 per transaction per day within a limit of R25,000 per applicant per calendar month.\n6. Substantiation of the original introduction of capital that has been introduced into South Africa for capital transfers.  \nReference:  \nSection B.5(A)(i) from Currency and Exchange Control Manual for Authorised Dealers"
        mock__get_api_response.return_value = path + api_response
        self.chat.user_provides_input(followup_content)
        messages = self.chat.format_messages_for_openai()        
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"].strip() == rag_system_response or "The documentation I have been provided does not help me answer the question. Please rephrase it and let's try again?"
        assert messages[-2]["role"] == "user"
        assert messages[-2]["content"].strip() == enhanced_user_content or 'Question: What documentation is required?\n\nExtract 1:\nI.1 Local financial assistance to affected persons and non-residents\n    (E) Non-residents\n        (i) Authorised Dealers may grant or authorise local financial assistance facilities to non-residents in respect of bona fide foreign direct investment in South Africa without restrictions, except where the funds are required for financial transactions and/or the acquisition of residential or commercial property in South Africa, the 1:1 ratio will apply.\n        (ii) As an exception to (i) above, Authorised Dealers may grant or authorise local financial assistance facilities to non-residents living and working in South Africa in respect of the acquisition of residential property, subject to normal lending criteria.\n        (iii) Any facility being made available to a non-resident party must be secured by an unencumbered Rand deposit or Rand based asset of equal or higher value. In addition, any facility accorded to the non-resident in respect of the aforementioned may not cause the borrower to exceed 100 per cent of the Rand value of funds introduced from abroad and invested locally.\n        (iv) If facilities are granted for the acquisition of fixed property, such facilities may not be increased at any stage based on a revaluation of the property in question.'


    @patch.object(CorpusChatCEMAD, '_get_api_response')
    def test_execute_path_no_retrieval_no_conversation_history(self, mock__get_api_response):
        self.chat.system_state = self.chat.State.RAG
        self.chat.reset_conversation_history()
        user_content = "Can I use my credit card online?"
        # mock a reasonable response from the LLM
        first_response = "Relevant"
        second_response = "Can an individual use their credit card for online purchases?|Can a company use its credit card for online purchases?"
        mock__get_api_response.side_effect = [first_response, second_response]

        self.chat.execute_path_no_retrieval_no_conversation_history(user_content)

        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["content"] == self.chat.Prefix.ALTERNATIVE.value + second_response
        assert self.chat.messages_intermediate[-2]["role"] == "user"
        assert self.chat.messages_intermediate[-2]["content"] == user_content


    @patch.object(CorpusChatCEMAD, '_get_api_response')
    def test_suggest_alternative_questions(self, mock__get_api_response):
        self.chat.system_state = self.chat.State.RAG
        self.chat.reset_conversation_history()
        user_content = "Can I use my credit card online?"

        # mock an empty response from the LLM
        mock__get_api_response.return_value = ""
        self.chat.suggest_alternative_questions(user_content)
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["content"] == self.chat.Errors.NO_DATA.value
        assert self.chat.messages_intermediate[-2]["role"] == "user"
        assert self.chat.messages_intermediate[-2]["content"] == user_content
        messages = self.chat.format_messages_for_openai()
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"].strip() == self.chat.Errors.NO_DATA.value.replace("ERROR: ", "")
        assert messages[-2]["role"] == "user"
        assert messages[-2]["content"].strip() == "Question: " + user_content

        api_response = "Can an individual use their credit card for online purchases?|Can a company use its credit card for online purchases?"
        mock__get_api_response.return_value = api_response
        self.chat.suggest_alternative_questions(user_content)
        assert self.chat.messages_intermediate[-1]["role"] == "assistant"
        assert self.chat.messages_intermediate[-1]["content"] == self.chat.Prefix.ALTERNATIVE.value + api_response
        assert self.chat.messages_intermediate[-2]["role"] == "user"
        assert self.chat.messages_intermediate[-2]["content"] == user_content



    def test_hardcode_response_for_question_not_relating_to_excon(self):
        user_content = "Hi"
        assistant_content = "I am a bot designed to answer questions about South African exchange control regulations. If you ask me a question about those, I will do my best to respond, with a reference. If I cannot find a relevant reference, I will not respond to the question rather than offing my opinion."
        self.chat.system_state = self.chat.State.RAG
        self.chat.reset_conversation_history()
        self.chat.hardcode_response_for_question_not_relating_to_excon(user_content)
        messages = self.chat.format_messages_for_openai()
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"].strip() == assistant_content
        assert messages[-2]["role"] == "user"
        assert messages[-2]["content"].strip() == "Question: " + user_content

    # def test_execute_path_no_retrieval_with_conversation_history(self):
    #     assert False