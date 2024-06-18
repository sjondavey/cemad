from cemad_rag.documents.cemad import CEMAD


from regulations_rag.corpus import Corpus , create_document_dictionary_from_folder


class CEMADCorpus(Corpus):
    def __init__(self, folder):
        document_dictionary = create_document_dictionary_from_folder(folder, globals())
        super().__init__(document_dictionary)
