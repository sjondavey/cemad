from cemad_rag.documents.cemad import CEMAD

import sys
sys.path.append('E:/Code/chat/gdpr')

class TestCEMAD:
    path_to_manual_as_csv_file = "./inputs/documents/ad_manual.csv"
    doc = CEMAD(path_to_manual_as_csv_file)

    def test_construction(self):
        assert True

    
    def test_get_text(self):
        response = self.doc.get_text('A.3(E)(viii)(a)(bb)', add_markdown_decorators = True, add_headings = True, section_only = False)
        expected_response = 'A.3 Duties and responsibilities of Authorised Dealers\n    (E) Transactions with Common Monetary Area residents\n        (viii) As an exception to (vi) above, Authorised Dealers may:\n            (a) sell foreign currency to:\n                (bb) CMA residents in South Africa, to cover unforeseen incidental costs whilst in transit, subject to viewing a passenger ticket confirming a destination outside the CMA;'
        assert response == expected_response


    def test_get_heading(self):
        response = self.doc.get_heading('A.3(E)(viii)(a)(bb)')
        expected_response = 'A.3 Duties and responsibilities of Authorised Dealers. (E) Transactions with Common Monetary Area residents.'
        assert response == expected_response




# class TestCEMADReferenceChecker:
#     path_to_manual_as_csv_file = "./inputs/documents/ad_manual.csv"
#     doc = CEMAD(path_to_manual_as_csv_file)

#     reference_checker = doc.reference_checker
