import pandas as pd

# Note: This method will not work correctly if empty values in the dataframe are NaN as is the case when loading
#       a dataframe form a file without the 'na_filter=False' option. You should ensure that the dataframe does 
#       not have any NaN value for the text fields. Try running df.isna().any().any() as a test before you get here
def get_regulation_detail(node_str, df, valid_index_tracker):
    # if not valid_index_tracker.is_valid_reference(node_str):
    #     return "The reference did not conform to this documents standard"
    text = ''
    terminal_text_df = df[df['full_reference'].str.startswith(node_str)]
    if len(terminal_text_df) == 0:
        return f"No section could be found with the reference {node_str}"
    terminal_text_index = terminal_text_df.index[0]
    terminal_text_indent = 0 # terminal_text_df.iloc[0]['Indent']
    for index, row in terminal_text_df.iterrows():
        number_of_spaces = (row['Indent'] - terminal_text_indent) * 4
        #set the string "line" to start with the number of spaces
        line = " " * number_of_spaces
        if pd.isna(row['Reference']) or row['Reference'] == '':
            line = line + row['Text']
        else:
            if pd.isna(row['Text']):
                line = line + row['Reference']
            else:     
                line = line + row['Reference'] + " " + row['Text']
        if text != "":
            text = text + "\n"
        text = text + line

    if node_str != '': #i.e. there is a parent
        parent_reference = valid_index_tracker.get_parent_reference(node_str)
        all_conditions = ""
        all_qualifiers = ""
        while parent_reference != "":
            parent_text_df = df[df['full_reference'] == parent_reference]
            conditions = ""
            qualifiers = ""
            for index, row in parent_text_df.iterrows():
                if index < terminal_text_index:
                    number_of_spaces = (row['Indent'] - terminal_text_indent) * 4
                    if conditions != "":
                        conditions = conditions + "\n"
                    conditions = conditions + " " * number_of_spaces
                    if (row['Reference'] == ''):
                        conditions = conditions + row['Text']
                    else:
                        conditions = conditions + row['Reference'] + " " +  row['Text']
                else:
                    number_of_spaces = (row['Indent'] - terminal_text_indent) * 4
                    if (qualifiers != ""):
                        qualifiers = qualifiers + "\n"
                    qualifiers = qualifiers + " " * number_of_spaces
                    if (row['Reference'] == ''):
                        qualifiers = qualifiers + row['Text']
                    else:
                        qualifiers = qualifiers + row['Reference'] + " " + row['Text']

            if conditions != "":
                all_conditions = conditions + "\n" + all_conditions
            if qualifiers != "":
                all_qualifiers = all_qualifiers + "\n" + qualifiers
            parent_reference = valid_index_tracker.get_parent_reference(parent_reference)

        if all_conditions != "":
            text = all_conditions +  text
        if all_qualifiers != "":
            text = text + all_qualifiers

    return text