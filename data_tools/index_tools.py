import numpy as np
import pandas as pd
from regulations_rag.embeddings import get_ada_embedding

def update_text_in_index(openai_client, index_df, text_to_change, changed_text, embedding_model, embedding_dimensions):
    existing_text = index_df['text'].tolist()
    if text_to_change not in existing_text:
        print(f"The text that needs changing is not in the input dataframe")
        return
    if len(index_df.iloc[0]['embedding']) != embedding_dimensions:
        print("The embedding_dimensions parameter does not match the embedding dimensions in the input DataFrame")
        return

    new_embedding = get_ada_embedding(openai_client = openai_client, text = changed_text, model = embedding_model, dimensions = embedding_dimensions)
    
    tmp = index_df[index_df['text'] == text_to_change]
    if len(tmp) > 1:
        print("The text_to_change value appears more than once in the input DataFrame")
        return
    target_index = tmp.index[0]

    index_df.at[target_index, 'text'] = changed_text
    index_df.at[target_index, 'embedding'] = np.array(new_embedding)
    return index_df


def add_to_index(openai_client, index_df, text, section_reference, source, embedding_model, embedding_dimensions, document):

    new_embedding = get_ada_embedding(openai_client = openai_client, text = text, model = embedding_model, dimensions = embedding_dimensions)

    new_df = pd.DataFrame([[section_reference, text, source, new_embedding, document]],columns = ["section_reference", "text", "source" , "embedding", "document"])
    index_df = pd.concat([index_df, new_df], ignore_index=True)
    return index_df


def remove_from_index(index_df, text_to_delete):
    existing_text = index_df['text'].tolist()
    if text_to_delete not in existing_text:
        print(f"The text that needs to be deleted is not in the input DataFrame")
        return

    if len(index_df[index_df['text'] == text_to_delete]) > 1:
        print("The text_to_delete appears multiple times in the DataFrame")
        return

    index_df = index_df[index_df['text'] != text_to_delete].reset_index(drop=True)
    return index_df



