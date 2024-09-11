import streamlit as st

if 'chat' not in st.session_state:
    st.switch_page('Ask_Questions.py')

st.title('Thanks for reading the instructions')


d = '''This Question Answering service is an example of **Retrieval Augmented Generation (RAG)**. It uses a Large Language Model to answer questions based on its reference material (see the Table of Contents page). To reduce the chance of incorrect answers, a key feature of the service is its ability to choose not to answer a question. However, there may be times when it doesn't answer a question that it should. In such cases, here are a few things you can try:

- **Specify the direction of currency flow**: Inflows and outflows are treated in different sections of CEMAD. If the question is ambiguous about the direction of the flow, the system may not retrieve the relevant documentation. For example, instead of asking "Who can trade gold?" try asking "Who can import gold?" or "Who can export gold?"

- **Ensure the question is complete**: If the question only makes sense in the context of the chat (e.g., "What is the BOP code for this?"), rephrase it as a complete question. For example, ask "What is the BOP code for [insert subject based on the conversation history]?"

- **Avoid specific country names**: CEMAD typically does not refer to countries other than South Africa by name. If your question includes a specific country name, change it to "foreign country" or "a member of the Common Monetary Area (CMA)." For example, "Can I open a non-resident rand account for an individual from Eswatini?" should be changed to "Can I open a non-resident rand account for an individual from the Common Monetary Area?"

- **Avoid specific currency names**: CEMAD generally doesn't reference specific currencies other than the Rand. Instead, it uses terms like "foreign currency" or "CMA country currency." For example, "Can I receive dividends in dollars?" should be changed to "Can I receive dividends in foreign currency?" or "Can I receive dividends in a CMA country currency?" since "dollars" could refer to US dollars or Namibian dollars.

- **Clarify the subject of the query**: There are different exchange control regulations and thresholds for individuals and companies. If the question doesn't make this distinction clear, add the necessary context. For example, "How much money can I invest offshore?" should be clarified as "How much money can an individual invest offshore?" or "How much money can a company invest offshore?"'''



st.markdown(d)