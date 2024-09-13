import streamlit as st


st.title('Documentation')

with st.sidebar:
    st.markdown('Thanks for reading the instructions. You are one of a small minority of people and you deserve a gold star!')


d = '''This Question Answering service is an example of **Retrieval Augmented Generation (RAG)**. It uses a Large Language Model to answer questions based on its reference material (which you can see the in the Table of Contents page). This service is not official nor endorsed by anyone relevant. Its answers should be treated as guidance, not law. If you use these answers as the basis to perform an action and that action is illegal, there is nobody to sue or join with you in your court case. You will be on your own, with only your blind faith in Large Language Models for company. 

To reduce the chance of incorrect answers, a key feature of this service is its ability not to answer when it cannot find relevant source material. There may be times when this feature feels more like a bug. In those cases, there are a few things you can try:

- **Specify the direction of currency flow**: Inflows and outflows are treated in different sections of CEMAD. If the question is ambiguous about the direction of the flow, the system may not retrieve the relevant documentation. For example, instead of asking "Who can trade gold?" try asking "Who can import gold?" or "Who can export gold?"

- **Ensure the question is complete**: If the question only makes sense in the context of the chat (e.g., "What is the BOP code for this?"), rephrase it as a complete question. For example, ask "What is the BOP code for gold imports?"

- **Avoid specific country names**: CEMAD typically does not refer to countries other than South Africa by name. If your question includes a specific country name, change it to "foreign country" or "a member of the Common Monetary Area (CMA)." For example, "Can I open a non-resident rand account for an individual from Eswatini?" should be changed to "Can I open a non-resident rand account for an individual from the Common Monetary Area?"

- **Avoid specific currency names**: CEMAD generally doesn't reference specific currencies other than the Rand. Instead, it uses terms like "foreign currency" or "CMA country currency." For example, "Can I receive dividends in dollars?" should be changed to "Can I receive dividends in foreign currency?" or "Can I receive dividends in a CMA country currency?" since "dollars" could refer to US dollars or Namibian dollars.

- **Clarify the subject of the query**: There are different exchange control regulations and thresholds for individuals and companies. If the question doesn't make this distinction clear, add the necessary context. For example, "How much money can I invest offshore?" should be clarified as "How much money can an individual invest offshore?" or "How much money can a company invest offshore?"

If you want to get some insight into how this app was built, have a look [here](https://www.aleph-one.co)
'''



st.markdown(d)