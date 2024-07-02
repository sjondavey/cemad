import os
import pandas as pd
from openai import OpenAI
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

    rerank_algo  = RerankAlgos.MOST_COMMON

    chat = CorpusChatCEMAD(openai_client = openai_client, 
                      embedding_parameters = embedding_parameters, 
                      chat_parameters = chat_parameters, 
                      corpus_index = corpus_index,
                      rerank_algo = rerank_algo,   
                      user_name_for_logging = 'test_user')


    def test_construction(self):
        assert True


    def test_reformat_assistant_answer(self):
        # if there are no sections in the rag, return the raw response 
        sections_in_rag = []
        df_search_sections = pd.DataFrame(columns = ["document", "section_reference"])
        definitions_in_rag = []
        df_definitions = pd.DataFrame(columns = ["document", "section_reference"])
        raw_response = "Some random text here."    
        result = {"success": True, "path": 'ANSWER:', "answer": raw_response, "reference": []}        
        formatted_response, dfns, sections = self.chat.reformat_assistant_answer(result, df_definitions=df_definitions, df_search_sections=df_search_sections)
        assert formatted_response == raw_response


        sections_in_rag = [["CEMAD", "E.(A)"], ["CEMAD", "B.12(A)"]]
        df_search_sections = pd.DataFrame(sections_in_rag, columns = ["document", "section_reference"])
        raw_response = "Some random text here."
        result = {"success": True, "path": 'ANSWER:', "answer": raw_response, "reference": [1]}        
        formatted_response, dfns, sections = self.chat.reformat_assistant_answer(result, df_definitions=df_definitions, df_search_sections=df_search_sections)
        assert formatted_response == 'Some random text here.  \nReference:  \nSection E.(A) from Currency and Exchange Control Manual for Authorised Dealers  \n'
        assert len(sections) == 1
        assert sections.iloc[0]['section_reference'] == 'E.(A)' 

        raw_response = "Some random text here."
        result = {"success": True, "path": 'ANSWER:', "answer": raw_response, "reference": [1, 2]}        
        formatted_response, dfns, sections = self.chat.reformat_assistant_answer(result, df_definitions=df_definitions, df_search_sections=df_search_sections)
        assert formatted_response == 'Some random text here.  \nReference:  \nSection E.(A) from Currency and Exchange Control Manual for Authorised Dealers  \nSection B.12(A) from Currency and Exchange Control Manual for Authorised Dealers  \n'
        assert len(sections) == 2
        assert sections.iloc[0]['section_reference'] == 'E.(A)' 
        assert sections.iloc[1]['section_reference'] == 'B.12(A)' 


    def test_append_content(self):
        self.chat.append_content('user', 'Question: What documents are required')
        assert len(self.chat.messages) == 1
        assert self.chat.messages[-1]['content'] == 'Question: What documents are required'
        assert self.chat.messages[-1]['role'] == 'user'

        assert len(self.chat.messages_without_rag) == 1
        assert self.chat.messages_without_rag[-1]['role'] == 'user'
        assert self.chat.messages_without_rag[-1]['content'] == 'What documents are required'

        # Try to add content for a role that does not exist
        self.chat.reset_conversation_history()
        self.chat.append_content('other_role', 'Question: What documents are required')
        assert len(self.chat.messages) == 0
        assert len(self.chat.messages_without_rag) == 0

        self.chat.reset_conversation_history()
        self.chat.append_content('assistant', 'Answer here')
        assert len(self.chat.messages) == 1
        assert self.chat.messages[-1]['content'] == 'Answer here'
        assert self.chat.messages[-1]['role'] == 'assistant'

        assert len(self.chat.messages_without_rag) == 1
        assert self.chat.messages_without_rag[-1]['role'] == 'assistant'
        assert self.chat.messages_without_rag[-1]['content'] == 'Answer here'


    def test_user_provides_input(self):
        # check the response if the system is stuck
        self.chat.system_state = self.chat.State.STUCK
        user_content = "How much money can an individual take offshore in any year?"
        self.chat.user_provides_input(user_content)
        assert self.chat.messages[-1]["role"] == "assistant"
        assert self.chat.messages[-1]["content"] == self.chat.Errors.STUCK.value
        assert self.chat.system_state == self.chat.State.STUCK

        # check the response if the system is in an unknown state
        self.chat.system_state = "random state not in list"
        user_content = "How much money can an individual take offshore in any year?"
        self.chat.user_provides_input(user_content)
        assert self.chat.messages[-1]["role"] == "assistant"
        assert self.chat.messages[-1]["content"] == self.chat.Errors.UNKNOWN_STATE.value

        # test the workflow if the system answers the question as hoped
        self.chat.system_state = self.chat.State.RAG
        user_content = "Who can trade gold?" # there are hits in the KB for this
        ### Prep for manual override
        testing = True # don't make call to openai API, use the canned response below        
        workflow_triggered, relevant_definitions, relevant_sections = self.chat.similarity_search(user_content)
        assert relevant_sections.iloc[0]["section_reference"] == "C.(C)" # need to confirm this will be returned in the "user_provides_input" function that uses the results of similarity_search 
        flag = "ANSWER:"
        # input_response = 'The acquisition of gold for legitimate trade purposes, such as by manufacturing jewellers or dentists, is subject to the approval of the South African Diamond and Precious Metals Regulator. After receiving such approval, a permit must be obtained from SARS, which will allow the permit holder to approach Rand Refinery Limited for an allocation of gold. The holders of gold, having received the necessary approvals, are exempt from certain provisions of Regulation 5(1).  \nReference:  \nC.(C)'
        input_response = 'The acquisition of gold for legitimate trade purposes, such as by manufacturing jewellers or dentists, is subject to the approval of the South African Diamond and Precious Metals Regulator. After receiving such approval, a permit must be obtained from SARS, which will allow the permit holder to approach Rand Refinery Limited for an allocation of gold. The holders of gold, having received the necessary approvals, are exempt from certain provisions of Regulation 5(1). Reference: 1'
        #response = {"success": True, "path": path, "answer": input_response, "reference": [1]}
        manual_responses_for_testing = [flag + input_response]

        # Return dictionaries:
        #{"success": False, "path": "SECTION:"/"ANSWER:", "llm_followup_instruction": llm_instruction} 
        #{"success": True, "path": "SECTION:", "document": 'GDPR', "section": section_reference}
        #{"success": True, "path": "ANSWER:"", "answer": llm_text, "reference": references_as_integers}
        #{"success": True, "path": "NONE:"}

        output_response = 'The acquisition of gold for legitimate trade purposes, such as by manufacturing jewellers or dentists, is subject to the approval of the South African Diamond and Precious Metals Regulator. After receiving such approval, a permit must be obtained from SARS, which will allow the permit holder to approach Rand Refinery Limited for an allocation of gold. The holders of gold, having received the necessary approvals, are exempt from certain provisions of Regulation 5(1).  \nReference:  \nSection C.(C) from Currency and Exchange Control Manual for Authorised Dealers'
        self.chat.user_provides_input(user_content,
                                       testing = testing,
                                       manual_responses_for_testing = manual_responses_for_testing)
        assert self.chat.messages[-1]["role"] == "assistant"
        assert self.chat.messages[-1]["content"].strip() == output_response
        assert self.chat.system_state == self.chat.State.RAG # rag

        # test the workflow if the system cannot find useful content in the supplied data
        self.chat.system_state = self.chat.State.RAG
        user_content = "Who can trade gold?" # there are hits in the KB for this
        testing = True # don't make call to openai API, use the canned response below
        flag = "NONE:"
        #response = {"success": True, "path": "NONE:"}
        manual_responses_for_testing = [flag + user_content]
        self.chat.user_provides_input(user_content,
                                       testing = testing,
                                       manual_responses_for_testing = manual_responses_for_testing)
        assert self.chat.messages[-1]["role"] == "assistant"
        assert self.chat.messages[-1]["content"].strip() == self.chat.Errors.NO_RELEVANT_DATA.value
        assert self.chat.system_state == self.chat.State.RAG

        # test the workflow if the system needs additional content
        self.chat.system_state = self.chat.State.RAG
        user_content = "Who can trade gold?" # there are hits in the KB for this
        testing = True # don't make call to openai API, use the canned response below
        #response = {"success": True, "path": "SECTION:", "document": 'CEMAD', "section": "A.3(A)(i)", "extract": 1}
        flag = "SECTION:"
        manual_responses_for_testing = [flag + "Extract 1, Reference C.(C)"]

        # now the response once it has received the additional data
        response_text = "The acquisition of gold for legitimate trade purposes, such as by manufacturing jewellers or dentists, is subject to the approval of the South African Diamond and Precious Metals Regulator. After receiving such approval, a permit must be obtained from SARS, which will allow the permit holder to approach Rand Refinery Limited for an allocation of gold. The holders of gold, having received the necessary approvals, are exempt from certain provisions of Regulation 5(1)."
        #response2 = {"success": True, "path": "ANSWER:", "answer": response_text, "reference": [1]}
        flag = "ANSWER:"
        manual_responses_for_testing.append(flag + response_text)

        self.chat.user_provides_input(user_content,
                                       testing = testing,
                                       manual_responses_for_testing = manual_responses_for_testing)
        assert self.chat.messages[-2]["role"] == "user"
        assert self.chat.messages[-2]["content"] == 'Question: Who can trade gold?\n\nExtract 1:\nC. Gold\n\n&nbsp;&nbsp;&nbsp;&nbsp;(C) Acquisition of gold for trade purposes\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(i) The acquisition of gold for legitimate trade purposes by e.g. manufacturing jewellers, dentists, is subject to the approval of the South African Diamond and Precious Metals Regulator.\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(ii) After receiving such approval, a permit must be obtained from SARS which will entitle the permit holder to approach Rand Refinery Limited for an allocation of gold.\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(iii) The holders of gold, having received the approvals outlined above, are exempt from the provisions of Regulation 5(1).\nExtract 2:\nC. Gold\n\n&nbsp;&nbsp;&nbsp;&nbsp;(B) Other exports of gold\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(i) All applications for permission to export gold in any form should be referred to the South African Diamond and Precious Metals Regulator.\n'

        assert self.chat.messages[-1]["role"] == "assistant"
        assert self.chat.messages[-1]["content"].strip() == response_text
        assert self.chat.system_state == self.chat.State.RAG

        # Test what happens if it calls for a section that it already has
        self.chat.reset_conversation_history()
        testing = True # don't make call to openai API, use the canned response below
        flag = "SECTION:"
        #response = {"success": True, "path": "SECTION:", "document": 'CEMAD', "section": "C.(C)", "extract": 1}
        manual_responses_for_testing = [flag + "Extract 1, Reference C.(C)"]


        response_text = "The acquisition of gold for legitimate trade purposes, such as by manufacturing jewellers or dentists, is subject to the approval of the South African Diamond and Precious Metals Regulator. After receiving such approval, a permit must be obtained from SARS, which will allow the permit holder to approach Rand Refinery Limited for an allocation of gold. The holders of gold, having received the necessary approvals, are exempt from certain provisions of Regulation 5(1). Reference: 1."
        # response2 = {"success": True, "path": "ANSWER:", "answer": response_text, "reference": [1]}
        manual_responses_for_testing.append(response_text)

        self.chat.user_provides_input(user_content,
                                       testing = testing,
                                       manual_responses_for_testing = manual_responses_for_testing)
        assert self.chat.messages[-2]["role"] == "user"
        assert self.chat.messages[-2]["content"] == 'Question: Who can trade gold?\n\nExtract 1:\nC. Gold\n\n&nbsp;&nbsp;&nbsp;&nbsp;(C) Acquisition of gold for trade purposes\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(i) The acquisition of gold for legitimate trade purposes by e.g. manufacturing jewellers, dentists, is subject to the approval of the South African Diamond and Precious Metals Regulator.\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(ii) After receiving such approval, a permit must be obtained from SARS which will entitle the permit holder to approach Rand Refinery Limited for an allocation of gold.\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(iii) The holders of gold, having received the approvals outlined above, are exempt from the provisions of Regulation 5(1).\nExtract 2:\nC. Gold\n\n&nbsp;&nbsp;&nbsp;&nbsp;(B) Other exports of gold\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(i) All applications for permission to export gold in any form should be referred to the South African Diamond and Precious Metals Regulator.\n'

        assert self.chat.messages[-1]["role"] == "assistant"
        assert self.chat.messages[-1]["content"].strip() == 'The acquisition of gold for legitimate trade purposes, such as by manufacturing jewellers or dentists, is subject to the approval of the South African Diamond and Precious Metals Regulator. After receiving such approval, a permit must be obtained from SARS, which will allow the permit holder to approach Rand Refinery Limited for an allocation of gold. The holders of gold, having received the necessary approvals, are exempt from certain provisions of Regulation 5(1).  \nReference:  \nSection C.(C) from Currency and Exchange Control Manual for Authorised Dealers'

        assert self.chat.system_state == self.chat.State.RAG


        # test what happens if the LLM does not listen to instructions and returns something random
        self.chat.system_state = self.chat.State.RAG
        user_content = "Who can trade gold?" # there are hits in the KB for this
        testing = True # don't make call to openai API, use the canned response below
        manual_responses_for_testing = ["None of the supplied documentation was relevant"]
        manual_responses_for_testing.append("None of the supplied documentation was relevant") # need to add it twice when checking this branch
        self.chat.user_provides_input(user_content, 
                                       testing = testing,
                                       manual_responses_for_testing = manual_responses_for_testing)
        assert self.chat.messages[-1]["role"] == "assistant"
        assert self.chat.messages[-1]["content"].strip() == self.chat.Errors.NOT_FOLLOWING_INSTRUCTIONS.value
        assert self.chat.system_state == self.chat.State.STUCK


    def test__add_rag_data_to_question(self):
        dfns = []
        dfns.append("def1")
        dfns.append("def2")
        df_definitions = pd.DataFrame(dfns, columns = ["definition"])
        sections = []
        section = "C.(C)(i)"
        sections.append([section, self.chat.corpus.get_text("CEMAD", section)])
        #sections.append("Z.2(A)(i)")
        df_search_sections = pd.DataFrame(sections, columns = ["section_reference", "regulation_text"])
        question = "user asks question"
        output_string = self.chat._add_rag_data_to_question(question, df_definitions, df_search_sections)

        expected_text = 'Question: user asks question\n\nExtract 1:\ndef1\nExtract 2:\ndef2\nExtract 3:\nC. Gold\n\n&nbsp;&nbsp;&nbsp;&nbsp;(C) Acquisition of gold for trade purposes\n\n&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(i) The acquisition of gold for legitimate trade purposes by e.g. manufacturing jewellers, dentists, is subject to the approval of the South African Diamond and Precious Metals Regulator.\n'

        assert output_string == expected_text

    def test__create_system_message(self):
        expected_message = "You are answering questions about South African 'Currency and Exchange Manual for Authorised Dealers' (CEMAD) for an Authorised Dealer (AD) based only on the reference extracts provided. You have 3 options:\n1) Answer the question. Preface an answer with the tag 'ANSWER:'. All referenced extracts must be quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword 'Reference: '. Do not include the word Extract, only provide the number(s).\n2) Request additional documentation. If, in the body of the extract(s) provided, there is a reference to another section that is directly relevant and not already provided, respond with the word 'SECTION:' followed by 'Extract extract_number, Reference section_reference' - for example SECTION: Extract 1, Reference [A-Z]\\.\\d{0,2}(?:\\([A-Z]\\))?(?:\\((?:i|ii|iii|iv|v|vi)\\))?(?:\\([a-z]\\))?(?:\\([a-z]{2}\\))?(?:\\(\\d+\\))?.\n3) State 'NONE:' and nothing else in all other cases\n"
        assert self.chat._create_system_message() == expected_message

        expected_message = "You are answering questions about South African 'Currency and Exchange Manual for Authorised Dealers' (CEMAD) for an Authorised Dealer (AD) based only on the reference extracts provided. You have 2 options:\n1) Answer the question. Preface an answer with the tag 'ANSWER:'. All referenced extracts must be quoted at the end of the answer, not in the body, by number, in a comma separated list starting after the keyword 'Reference: '. Do not include the word Extract, only provide the number(s).\n2) State 'NONE:' and nothing else in all other cases\n"
        assert self.chat._create_system_message(number_of_options=2) == expected_message



    def test_resource_augmented_query(self):
        self.chat.reset_conversation_history()
        user_content = "Who can trade gold?"
        # Add a message to the queue and give the LLM relevant data from which to answer the question
        self.chat.system_state = self.chat.State.RAG
        # NOTE: I am not going to test the openai api call. I am going to use 'testing' mode with canned answers
        testing = True

        wf, dfns, sections = self.chat.similarity_search(user_content)
        response_text = "test to see what happens when if the API believes it successfully answered the question with the resources provided"
        response2 = {"success": True, "path": "ANSWER:", "answer": response_text, "reference": [1]}
        manual_responses_for_testing = []
        manual_responses_for_testing.append(response2)

        # Check that if the question and reference data mismatch, the system returns a NONE: value
        user_content = "How much money can an individual take offshore in any year?"

        testing = True
        # manual_responses_for_testing = []
        # manual_responses_for_testing.append("NONE: test to see what happens when if the API believes it cannot answer the question with the resources provided")
        response = self.chat.resource_augmented_query(user_question = user_content, df_definitions = dfns, 
                                                                df_search_sections = sections)
        assert response["path"] == self.chat.Prefix.NONE.value
        
        

    def test_similarity_search(self):
        if self.include_calls_to_api:
            # Check that random chit-chat to the main dataset does not return any hits from the embeddings
            text = "Hi"
            workflow_triggered, df_definitions, df_search_sections = self.chat.similarity_search(text)
            assert len(df_definitions) == 0
            assert len(df_search_sections) == 0 
            # now move to the testing dataset for fine grained tests
            user_content = "Who can trade gold?"
            workflow_triggered, relevant_definitions, relevant_sections = self.chat.similarity_search(user_content)
            assert len(relevant_definitions) == 0
            assert len(relevant_sections) == 2
            assert relevant_sections.iloc[0]["section_reference"] == 'C.(C)'
            assert relevant_sections.iloc[1]["section_reference"] == 'C.(B)'




    def test_add_section_to_resource(self):
        # Note that I need to use references that appear in the test data
        text = "A.3 Duties and responsibilities of Authorised Dealers\n\
                    (A) Introduction\n\
                        (i) Fake reference to B.4(B)(iv)(f)"
        manual_data = []
        manual_data.append(["CEMAD", "A.3(A)(i)", 0.15, 1, text, 100])
        df_manual_data = pd.DataFrame(manual_data, columns = ["document", "section_reference", "cosine_distance", "count", "regulation_text", "token_count"])
        df_definitions = pd.DataFrame(columns = ["document", "section_reference"])
        result = {"success": True, "path": "SECTION:", "extract": 1, "document": "CEMAD", "section": "B.4(B)(iv)(f)"} # NB the document may not be the same as the document in extract_num_as_int
        df_updated = self.chat.add_section_to_resource(result, df_definitions, df_search_sections = df_manual_data)
        assert len(df_updated) == 2
        assert df_updated.iloc[0]['section_reference'] == "A.3(A)(i)"
        assert df_updated.iloc[1]['section_reference'] == 'B.4(B)(iv)(f)'

        # Add a second reference to the RAG data
        text_2 = "A.3 Duties and responsibilities of Authorised Dealers\n\
                    (A) Introduction\n\
                        (ii) No references to be found here"
        manual_data.append(["CEMAD", "A.3(A)(ii)", 0.14, 1, text_2, 100])
        df_manual_data = pd.DataFrame(manual_data, columns = ["document", "section_reference", "cosine_distance", "count", "regulation_text", "token_count"])
        result = {"success": True, "path": "SECTION:", "extract": 1, "document": "CEMAD", "section": "B.4(B)(iv)(f)"} # NB the document may not be the same as the document in extract_num_as_int
        df_updated = self.chat.add_section_to_resource(result, df_definitions, df_search_sections = df_manual_data)

        assert len(df_updated) == 3
        assert df_updated.iloc[0]['section_reference'] == "A.3(A)(i)"
        assert df_updated.iloc[1]['section_reference'] == 'A.3(A)(ii)'
        assert df_updated.iloc[2]['section_reference'] == 'B.4(B)(iv)(f)'

        # Add a third reference to the RAG data
        text_3 = "A.3 Local facilities to non-residents\n\
                    (B) Random Text with another reference to B.4(B) (iv) (f) but with some random spaces"
        manual_data.append(["CEMAD", "A.3(B)", 0.13, 1, text_3, 100])        
        df_manual_data = pd.DataFrame(manual_data, columns = ["document", "section_reference", "cosine_distance", "count", "regulation_text", "token_count"])
        result = {"success": True, "path": "SECTION:", "extract": 1, "document": "CEMAD", "section": "B.4(B)(iv)(f)"} # NB the document may not be the same as the document in extract_num_as_int
        df_updated = self.chat.add_section_to_resource(result, df_definitions, df_search_sections = df_manual_data)
        assert len(df_updated) == 4
        assert df_updated.iloc[0]['section_reference'] == "A.3(A)(i)"
        assert df_updated.iloc[1]['section_reference'] == 'A.3(A)(ii)'
        assert df_updated.iloc[2]['section_reference'] == "A.3(B)"
        assert df_updated.iloc[3]['section_reference'] == 'B.4(B)(iv)(f)'



    def test__truncate_message_list(self):
        l = [{"content": "1"}, 
            {"content": "2"},
            {"content": "3"},
            {"content": "4"},
            {"content": "5"},
            {"content": "6"},
            {"content": "7"},
            {"content": "8"},
            {"content": "9"},
            {"content": "10"}]
        system_message = [{"content" : "s"}]
        truncated =self.chat._truncate_message_list(system_message, l, 2)
        assert len(truncated) == 2
        assert truncated[0]["content"] == "s"
        assert truncated[1]["content"] == "10"

        truncated =self.chat._truncate_message_list(system_message, l, 6)
        assert len(truncated) == 5
        assert truncated[0]["content"] == "s"
        assert truncated[1]["content"] == "7"
        assert truncated[4]["content"] == "10"

#     def test_enrich_user_request_for_documentation(self):
#         messages_without_rag = [{'role': 'user', 'content': 'Can foreign nationals send money home?'},
#                                 {'role': 'assistant', 'content': 'Yes, foreign nationals can send money abroad if they meet certain conditions. Foreign nationals temporarily in South Africa are required to declare whether they are in possession of foreign assets upon arrival. If they complete the necessary declarations and undertakings, they may be permitted to conduct their banking on a resident basis, dispose of or invest their foreign assets, conduct non-resident or foreign currency accounts, and transfer funds abroad. However, they must be able to substantiate the source of the funds and the value of the funds should be reasonable in relation to their income generating activities in South Africa. The completed declarations and undertakings must be retained by the Authorised Dealers for a period of five years. There are also exemptions for single remittance transactions up to R5,000 and transactions where a business relationship has been established. (B.5(A)(i)(d), B.5(A)(i)(e))'}]
#         user_content = 'Is there any documentation required?'
#         model_to_use = "gpt-3.5-turbo"
#         response = self.chat.enrich_user_request_for_documentation(user_content, messages_without_rag, model_to_use)
#         print(response)
#         assert(response.startswith('What documentation is required as evidence for'))

