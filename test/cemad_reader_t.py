import pytest
import pandas as pd

from cemad_rag.cemad_reference_checker import CEMADReferenceChecker

from cemad_rag.cemad_reader import  CEMADReader


#reference_checker = CEMADReferenceChecker()

#df = pd.read_csv("./inputs/ad_manual.csv", sep="|", encoding="utf-8", na_filter="")
#df = load_regulation_data_from_files("./inputs/ad_manual.csv", "./inputs/ad_manual_plus.csv")
test_reader = CEMADReader()

def test_get_regulation_detail():
    response = test_reader.get_regulation_detail('A.3(E)(viii)(a)(bb)')
    expected_response = 'A.3 Duties and responsibilities of Authorised Dealers\n    (E) Transactions with Common Monetary Area residents\n        (viii) As an exception to (vi) above, Authorised Dealers may:\n            (a) sell foreign currency to:\n                (bb) CMA residents in South Africa, to cover unforeseen incidental costs whilst in transit, subject to viewing a passenger ticket confirming a destination outside the CMA;'
    assert response == expected_response


def test_get_regulation_heading():
    response = test_reader.get_regulation_heading('A.3(E)(viii)(a)(bb)')
    expected_response = 'A.3 Duties and responsibilities of Authorised Dealers. (E) Transactions with Common Monetary Area residents.'
    assert response == expected_response

