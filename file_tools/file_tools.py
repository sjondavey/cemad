import pandas as pd
import os
import re
from cemad_rag.cemad_reference_checker import CEMADReferenceChecker



def process_regulations(filenames_as_list, reference_checker, non_text_labels):
    all_data_as_lines = []
    non_text = {}
    for i in range(0, len(non_text_labels)):
        non_text[non_text_labels[i]] = {}

    for file in filenames_as_list:
        #print(f"Processing file: {file}")
        text = {}
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()

        remaining_lines = text.split('\n')
        # remove empty lines
        remaining_lines = [line for line in remaining_lines if line.strip() != '']

        for i in range(0, len(non_text_labels)):
            remaining_lines, non_text_table_dict = extract_non_text(remaining_lines, '#' + non_text_labels[i])
            #print(non_text_table_dict)
            non_text[non_text_labels[i]].update(non_text_table_dict)

        # print(f'read {len(remaining_lines)} lines from {file})')
        all_data_as_lines.extend(remaining_lines)
        #print(f"Added: {len(remaining_lines)} lines")

    df = pd.DataFrame()
    df = process_lines(all_data_as_lines, reference_checker)
    add_section_reference(df, reference_checker) # adds the reference to the input dataframe
    df['word_count'] = df['text'].str.split().str.len()
    return df, non_text


def parse_line_of_text(line_of_text, reference_checker):
    """
    Parses a line of text to extract the indent level (as a multiple of 4 spaces), the index, and the remaining text.
    Validates the indent level and ensures the extracted index matches the expected regex pattern based on the indent level.

    Parameters:
        line_of_text (str): The line of text to be parsed.

    Returns:
        tuple: A tuple containing the indent level (number of spaces at the start of the line modulo 4), the index (if any), and the remaining text.

    Raises:
        ValueError: If the indent is not a multiple of 4, if the index is not appropriate for the indent level,
                    or if the index does not match the expected regex pattern.
    """
    stripped_line = line_of_text.lstrip(' ')
    indent = len(line_of_text) - len(stripped_line)
    if indent % 4 != 0:
        raise ValueError(f"This line does not have an indent which is a multiple of 4: {line_of_text}")
    indent = indent // 4

    index, remaining_text = reference_checker._extract_reference_from_string(stripped_line)

    if index: 
        if index in reference_checker.exclusion_list:
            if indent != 0:
                raise ValueError(f"This line has {indent} indent(s) but should have zero because the index is on the exclusion list")
            return indent, index, remaining_text

        if indent >= len(reference_checker.index_patterns):
            raise ValueError(f"This line has too many indents and cannot be compared against a Valid Index: {line_of_text}")

        expected_pattern = reference_checker.index_patterns[indent]
        match = re.match(expected_pattern, index)
        if not match:
            raise ValueError(f"This line has {indent} indent(s) and its index should match a regex pattern {expected_pattern} but it does not: {line_of_text}")

    return indent, index, remaining_text


def process_lines(lines, reference_checker):
    data = {
        "indent": [],
        "reference": [],
        "text": [],
        "document": [],
        "page": [],
        "heading": []
    }

    for line_of_text in lines:
        if line_of_text.strip() != '':  # Skip blank lines
            # Find and remove any special markup characters from the line of text
            pattern = r'\(([^\(\)]*\.pdf); pg (\d+)\)\s*$'
            # Use search to find matches
            document_page = re.search(pattern, line_of_text)
            if document_page:
                data['document'].append(document_page.group(1).strip())
                data['page'].append(document_page.group(2).strip())
                line_of_text = line_of_text[:document_page.start()] + line_of_text[document_page.end():]
            else:
                data['document'].append('')
                data['page'].append('')

            pattern = '\(#Heading\)'
            # Use search to find matches
            document_page = re.search(pattern, line_of_text)
            if document_page:
                data['heading'].append(True)
                line_of_text = line_of_text[:document_page.start()] + line_of_text[document_page.end():]
            else:
                data['heading'].append(False)

            #Now strip out the index part
            indent, reference, remaining_text = parse_line_of_text(line_of_text, reference_checker)

            data['indent'].append(indent)
            data['reference'].append(reference.strip())
            data['text'].append(remaining_text.strip())

    df = pd.DataFrame(data)
    return df


# TODO: Remove the page reference from the 'block_identifier' because this is messing up the dictionary keys
def extract_non_text(lines, block_identifier, hard_stop = 100):
    """
    Very crude function to extract the following blocks from the text:
            - block_identifier = 'Table', 'Formula' or 'Example'
    from the text
    """
    dictionary = {}
    current_block = None
    line_counter = 0
    remaining_lines = []

    
    for line in lines:
        stripped_line = line.lstrip(' ')
        if stripped_line.startswith(block_identifier):
            if '- end' in line:
                current_block = None
                line_counter = 0
            else:
                current_block = stripped_line.strip()
                dictionary[current_block] = []
            continue

        if current_block is not None:
            if line_counter < hard_stop:
                #dictionary[current_block].append(stripped_line)
                dictionary[current_block].append(line)
                line_counter += 1
            else:
                raise ValueError(f'Formatting issue with {current_block}: more than {hard_stop} lines before finding closing token: "{block_identifier} - end".')
        else:
            remaining_lines.append(line)
    
    if current_block is not None:
        raise ValueError(f'Formatting issue with {current_block}: reached the end of the input lines before finding closing token: "{block_identifier} - end".')

    return remaining_lines, dictionary



def add_section_reference(df, reference_checker):
    """
    This function adds a 'section_reference' column to a DataFrame. The 'section_reference' is calculated based on
    the values in the 'indent' and 'reference' columns using the following logic:

    1) If 'indent' is 0:
       - and 'reference' is not blank, 'section_reference' is 'reference';
       - and 'reference' is blank, 'section_reference' is the 'reference' from the last row with 'indent' 0
         that has a non-blank 'reference'.

    2) If 'indent' is > 0:
       - Calculate a stub reference which is equal to the value in 'reference' enclosed in round brackets if there
         is a value in 'reference'; otherwise, it is the 'reference' from the last row with the same 'indent'
         value that has a non-blank 'reference', enclosed in round brackets.
       - Calculate the root reference which is the 'section_reference' from the last row with 'indent' equal to
         this row's 'indent' - 1.
       - 'section_reference' is a concatenation of the root reference and the stub reference.
    """
    df['section_reference'] = ''

    for i in range(df.shape[0]):
        if df.loc[i, 'indent'] == 0:
            if pd.notna(df.loc[i, 'reference']) and df.loc[i, 'reference'].strip() != '':
                #df.loc[i, 'section_reference'] = '(' + df.loc[i, 'reference'].strip() + ')'
                df.loc[i, 'section_reference'] = df.loc[i, 'reference'].strip()
            else:
                ref = df.loc[:i, :].loc[(df['indent'] == 0) & (df['reference'].str.strip() != ''), 'reference'].values
                df.loc[i, 'section_reference'] = ref[-1] if ref.size > 0 else ''
        else:
            if pd.notna(df.loc[i, 'reference']) and df.loc[i, 'reference'].strip() != '':
                stub = df.loc[i, 'reference']
            else:
                ref = df.loc[:i, :].loc[(df['indent'] == df.loc[i, 'indent']) & (df['reference'].str.strip() != ''), 'reference'].values
                stub = ref[-1] if ref.size > 0 else ''

            root_ref = df.loc[:i, :].loc[df['indent'] == df.loc[i, 'indent'] - 1, 'section_reference'].values
            root = root_ref[-1] if root_ref.size > 0 else ''
            
            full_reference = root + stub
            if not reference_checker.is_valid(full_reference):
                raise ValueError(f'Unable to construct a valid full reference for line: {df.loc[i, "text"]}')

            df.loc[i, 'section_reference'] = full_reference



def read_processed_regs_into_dataframe(file_list, reference_checker, non_text_labels, print_summary = False):
    df, non_text = process_regulations(file_list, reference_checker, non_text_labels)
    #TODO: Remove the page numbers from the non-text keys
    if print_summary:
        print("total lines in dataframe: ", len(df))
        for key in non_text.keys():
            print("total ", key, ": ", len(non_text[key]))
    return df, non_text


