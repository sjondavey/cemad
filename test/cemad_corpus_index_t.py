import os
from openai import OpenAI

from regulations_rag.regulation_index import EmbeddingParameters
from regulations_rag.embeddings import get_ada_embedding
from regulations_rag.rerank import RerankAlgos

from cemad_rag.cemad_corpus_index import CEMADCorpusIndex

class TestCEMADCorpusIndex():
    key = os.getenv('excon_encryption_key')
    index = CEMADCorpusIndex(key)
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
    embedding_parameters = EmbeddingParameters("text-embedding-3-large", 1024)

    def test_construction(self):
        assert True

    def test_get_relevant_definitions(self):
        user_content = "what is offshoring?"
        user_content_embedding = get_ada_embedding(self.openai_client, user_content, self.embedding_parameters.model, self.embedding_parameters.dimensions)       
        relevant_definitions = self.index.get_relevant_definitions(user_content = user_content, user_content_embedding = user_content_embedding, threshold = self.embedding_parameters.threshold) 
        assert len(relevant_definitions) == 1
        definition = relevant_definitions.iloc[0]["definition"]
        expected_definition = 'Offshoring is the transferring of the business processes (including, but not limited to exchange control compliance), services, systems, data or infrastructure of the reporting entities to a branch or Head Office situated outside the borders of South Africa.'
        assert definition == expected_definition 

    def test_get_relevant_sections(self):
        user_content = "What is merchanting trade?"
        user_content_embedding = get_ada_embedding(self.openai_client, user_content, self.embedding_parameters.model, self.embedding_parameters.dimensions)     
        rerank_algo = RerankAlgos.NONE  
        relevant_sections = self.index.get_relevant_sections(user_content = user_content, user_content_embedding = user_content_embedding, threshold = self.embedding_parameters.threshold, rerank_algo=rerank_algo) 
        assert len(relevant_sections) == 3
        documents_that_are_referenced = relevant_sections['document'].unique()
        assert "CEMAD" in documents_that_are_referenced
        article = relevant_sections.iloc[2]["regulation_text"]
        expected_article = 'B.12 Merchanting, barter and counter trade\n    (B) Barter and counter trade\n        (i) Transactions of this nature must be referred to the Financial Surveillance Department for prior written approval.\n        (ii) Requests for barter and counter trade must be supported by copies of the contracts entered into between the relative parties with a full explanation of the manner in which the values of the goods have been arrived at. Where an open market or world price exists, any deviation therefrom must be fully substantiated and motivated.'
        assert article == expected_article
