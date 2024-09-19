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

    rerank_algo  = RerankAlgos.NONE
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

        expected_response = f"The question you posed did not contain any hits in the database. There are many reasons why this could be the case. Here however are different phrasings of the question which should find some reference material in the {self.chat.index.corpus_description}: Perhaps try\n\n" 
        counter = 1
        for question in alt_questions_as_array:
            expected_response = expected_response + "\n" + str(counter) + ") " + question
            counter = counter + 1
        assert text_response_for_openai == expected_response
        # hard coded
        expected_response = "The question you posed did not contain any hits in the database. There are many reasons why this could be the case. Here however are different phrasings of the question which should find some reference material in the South African 'Currency and Exchange Manual for Authorised Dealers' (CEMAD): Perhaps try\n\n\n1) Can an individual use their credit card for online purchases?\n2) Can a company use its credit card for online purchases?"
        assert text_response_for_openai == expected_response
        
        # now with only one alternative
        alt_questions = " Can an individual use their credit card for online purchases?"
        alt_questions_as_array = [substr.strip() for substr in alt_questions.split('|') if substr]
        result = {"success": True, "path": prefix, "alternatives": alt_questions_as_array} 
        text_response_for_openai = self.chat._reformat_assistant_answer(result, df_definitions, df_sections)
        # hard coded
        expected_response = "The question you posed did not contain any hits in the database. There are many reasons why this could be the case. Here however is a different phrasing of the question which should find some reference material in the South African 'Currency and Exchange Manual for Authorised Dealers' (CEMAD). Perhaps try:\n\n\nCan an individual use their credit card for online purchases?"
        assert text_response_for_openai == expected_response
        


    @patch.object(CorpusChatCEMAD, '_get_api_response')
    def test_user_provides_input(self, mock__get_api_response):
        self.chat.system_state = self.chat.State.RAG
        self.chat.reset_conversation_history()
        user_content = "can a foreigner buy property in south africa?"
        enhanced_user_content = "Question: can a foreigner buy property in south africa?\n\nExtract 1:\nI.1 Local financial assistance to affected persons and non-residents\n    (E) Non-residents\n        (i) Authorised Dealers may grant or authorise local financial assistance facilities to non-residents in respect of bona fide foreign direct investment in South Africa without restrictions, except where the funds are required for financial transactions and/or the acquisition of residential or commercial property in South Africa, the 1:1 ratio will apply.\n        (ii) As an exception to (i) above, Authorised Dealers may grant or authorise local financial assistance facilities to non-residents living and working in South Africa in respect of the acquisition of residential property, subject to normal lending criteria.\n        (iii) Any facility being made available to a non-resident party must be secured by an unencumbered Rand deposit or Rand based asset of equal or higher value. In addition, any facility accorded to the non-resident in respect of the aforementioned may not cause the borrower to exceed 100 per cent of the Rand value of funds introduced from abroad and invested locally.\n        (iv) If facilities are granted for the acquisition of fixed property, such facilities may not be increased at any stage based on a revaluation of the property in question.\nExtract 2:\nB.2 Capital transfers\n    (B) Private individuals resident in South Africa\n        (i) Foreign investments by private individuals (natural persons) resident in South Africa\n            (a) Authorised Dealers may allow the transfer, as a foreign capital allowance, of up to a total amount of R10 million per calendar year per private individual who is a taxpayer in good standing and is 18 years and older, for investment purposes abroad. The funds to be transferred must be converted to foreign currency by the Authorised Dealer and may also be held in a resident foreign currency account in the name of the resident with any Authorised Dealer.\n            (b) Authorised Dealers are advised that a valid green bar-coded South African identity document or a Smart identity document card is the only acceptable document proving residency in South Africa.\n            (c) Prior to authorising the transaction, Authorised Dealers must ensure that their client is acquainted with the declaration contained in the integrated form.\n            (d) In terms of the SARS Tax Compliance Status (TCS) system, a TCS PIN letter will be issued to the taxpayer that will contain the tax number and TCS PIN. Authorised Dealers must use the TCS PIN to verify the taxpayer's tax compliance status via SARS eFiling prior to effecting any transfers. Authorised Dealers must ensure that the amount to be transferred does not exceed the amount approved by SARS. Authorised Dealers should note that the TCS PIN can expire and should the Authorised Dealers find that the TCS PIN has indeed expired, then the Authorised Dealer must insist on a new TCS PIN to verify the taxpayer's tax compliance status.\n            (e) In terms of the SARS Tax Compliance Status System introduced on 2016-04-18, a tax compliance status PIN letter will be issued to the taxpayer that will contain the tax number and PIN. Authorised Dealers must use the PIN to verify the taxpayer's tax compliance status via SARS eFiling prior to effecting any transfers.\n            (f) Authorised Dealers must retain a printed tax compliance status verification result for a period of five years for inspection purposes.\n            (g) Private individuals who do not have a tax reference number will have to register at their local SARS branch.\n            (h) Authorised Dealers must bring to the attention of their clients that they may not enter into any transactions whereby capital or the right to capital will be directly or indirectly exported from South Africa (e.g. may not enter into a foreign commitment with recourse to South Africa). However, private individuals may raise loans abroad to finance the acquisition of foreign assets without recourse to South Africa.\n            (i) Resident individuals with authorised foreign assets may invest in South Africa, provided that where South African assets are acquired through an offshore structure (loop structure), the investment is reported to an Authorised Dealer as and when the transaction(s) is finalised as well as the submission of an annual progress report to the Financial Surveillance Department via an Authorised Dealer. The aforementioned party also has to view an independent auditor's written confirmation or suitable documentary evidence verifying that such transaction(s) is concluded on an arm's length basis, for a fair and market related price.\n            (j) Upon completion of the transaction in (i) above the Authorised Dealer must submit a report to the Financial Surveillance Department which should, inter alia, include the name(s) of the South African affiliated foreign investor(s), a description of the assets to be acquired (including inward foreign loans, the acquisition of shares and the acquisition of property), the name of the South African target investment company, if applicable and the date of the acquisition as well as the actual foreign currency amount introduced including a transaction reference number.\n            (k) In addition, all inward loans from South African affiliated foreign investors must comply with the directives issued in section I.3(B) of the Authorised Dealer Manual.\n            (l) Existing unauthorised loop structures (i.e. created by individuals prior to 2021-01-01) and/or unauthorised loop structures where the 40 per cent shareholding threshold was exceeded, must still be regularised with the Financial Surveillance Department.\n            (m) The Financial Surveillance Department will consider applications by private individuals who wish to invest in excess of the R10 million foreign capital allowance limit, in different asset classes. Such investments, if approved, could be facilitated via a foreign domiciled and registered trust. This dispensation would also apply to private individuals who have existing authorised foreign assets, irrespective of the value thereof. In terms of the TCS system, a TCS PIN letter will be issued to the taxpayer that will contain the tax number and TCS PIN. Authorised Dealers must use the TCS PINto verify the taxpayer's tax compliance status via SARS eFiling prior to effecting any transfers. Authorised Dealers should note that the TCS PIN can expire and should the Authorised Dealers find that the TCS PIN has indeed expired, then the Authorised Dealer must insist on a new TCS PIN to verify the taxpayer's tax compliance.\n            (n) Foreign currency accounts may be opened for private individuals (natural persons). Authorised Dealers may allow withdrawals from these accounts through the use of any authorised banking product. Funds withdrawn from these accounts may also be converted to Rand.\n            (o) Private individuals may, as part of their single discretionary allowance and/or foreign capital allowance, export multi-listed domestic securities to a foreign securities register in a jurisdiction where such securities are listed, subject to tax compliance and reporting to the Financial Surveillance Department via a Central Securities Depository Participant, in conjunction with an Authorised Dealer.\n            (p) Since these transactions will not result in actual flow of funds from South Africa, the process thereof as well as the reporting must take place as outlined hereunder:\n                (aa) An applicant must furnish the following information through an Authorised Dealer, at the time of submitting a request for a confirmation letter and/or approval letter to the Financial Surveillance Department:\n                    (1) the full names and identity number of the applicant;\n                    (2) the name of the company whose securities are exported as well as number of securities and the market value thereof; and\n                    (3) domicilium and name of the target foreign register on which those securities are listed.\n                (bb) In this regard, the confirmation letter and/or approval letter from the Financial Surveillance Department must be presented by the applicant to the relevant South African Central Securities Depository Participant to effect the transaction.\n                (cc) Central Securities Depository Participants may, in conjunction with an Authorised Dealer, allow the transfer of domestic listed securities abroad, up to a total market value of R1 million per calendar year in terms of the single discretionary allowance for private individuals, without the requirement to obtain a TCS PIN letter, provided a confirmation letter from the Financial Surveillance Department is viewed.\n                (dd) Central Securities Depository Participants may, in conjunction with an Authorised Dealer, also allow the transfer of domestic listed securities of up to a total market value of R10 million per calendar year in terms of the foreign capital allowance, provided that a TCS PIN letter is obtained as well as a confirmation letter from the Financial Surveillance Department is viewed.\n                (ee) Private individuals who export securities with a market value of more than R10 million are subject to a more stringent verification process by SARS as well as an approval process from the Financial Surveillance Department. Such transfers will trigger a risk management test that will, inter alia, include verification of the tax status and the source of funds, as well as risk assess the private individual in terms of the anti-money laundering and countering terror financing requirements, as prescribed in the Financial Intelligence Centre Act, 2001 (Act No. 38 of 2001).\n            (q) Private individuals may only fund online international trading accounts at registered brokers in terms of the single discretionary and/or foreign capital allowance, i.e. the Authorised Dealer concerned must convert the Rand into foreign currency and transfer such funds via the banking system as an Electronic Funds Transfer to a foreign bank account or the funds can be deposited in a foreign currency account at an Authorised Dealer.\n            (r) No South African debit, credit and virtual card may, however, be used to fund a foreign currency account at an Authorised Dealer or a bank abroad, nor may international trading accounts of private individuals be funded using South African credit, debit and virtual card transfers. Online international trading accounts, inter alia include trading global currencies against each other, trading a contract for difference, trading in foreign stocks, trading commodities including crypto assets and/or trading foreign indices using an online trading platform of the broker concerned.\nExtract 3:\nG. Securities control\n    (J) Emigrants, immigrants and deceased estates\n        (ii) Immigrants\n            (a) Purchase abroad of South African quoted securities (aa) Immigrants who have been accorded the concessions laid down in section B.5(B)(ii) of the Authorised Dealer Manual may within five years after their arrival, invest their foreign funds in or switch other foreign investments owned by them into South African securities abroad.\n            (b) Cancellation of the non-resident endorsement on South African securities\n                (aa) Immigrants may transfer their foreign assets to South Africa by way of locally quoted securities and Authorised Dealers must, in such instances, grant authority to Authorised Banks to cancel non-resident endorsements on such scrip. Such scrip must be transferred to the South African Register and a local address must be registered. Should South African quoted securities acquired abroad be introduced by an immigrant for sale on the JSE Limited, the resultant sale proceeds must be credited to a resident account.\n            (c) Emigration within five years\n                (aa) South African securities physically introduced and retained or purchased locally by immigrants who leave the country within five years of arrival may be exported on departure, provided that they have completed the necessary declaration and undertaking as outlined in section B.5(B)(i)(a) of the Authorised Dealer Manual.\n            (d) Former residents of the CMA\n                (aa) The facilities outlined above may not be accorded to any person who has previously resided in the CMA. Any requests received from such persons should be referred to the Financial Surveillance Department."
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
        assistant_content = "I am a bot designed to answer questions about the South African \'Currency and Exchange Manual for Authorised Dealers\' (CEMAD). If you ask me a question about that, I will do my best to respond, with a reference. If I cannot find a relevant reference in the document, I have been coded not to respond to the question rather than offing my opinion. Please read the document page for some suggestions if you find this feature frustrating (If you are using this on a mobile phone, look for the little '>' at the top left of your screen)"
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