import pandas as pd
import re
from regulations_rag.document import Document
from regulations_rag.file_tools import  load_csv_data
from regulations_rag.reference_checker import ReferenceChecker
from regulations_rag.regulation_table_of_content import StandardTableOfContent

from cemad_rag.cemad_reference_checker import CEMADReferenceChecker

class CEMAD_User_Queries(Document):
    def __init__(self, path_to_manual_as_csv_file = "./inputs/documents/ad_manual_plus.csv"):
        reference_checker = CEMADReferenceChecker()


        self.document_as_df = load_csv_data(path_to_file = path_to_manual_as_csv_file)

        document_name = "User queries about the Currency and Exchange Control Manual for Authorised Dealers"
        super().__init__(document_name, reference_checker=reference_checker)
        if not self.check_columns():
            raise AttributeError(f"The input csv file for the CEMAD User Query class does not have the correct column headings")

        self.toc = self.get_toc()

    def check_columns(self):
        ''' 
            'indent'            : number of indents before the line starts - to help interpret it (i) is the letter or the Roman numeral (for example)
            'reference'         : the part of the section_reference at the start of the line. Can be blank
            'text'              : the text on the line excluding the 'reference' and any special text (identifying headings, page number etc)
            'heading'           : boolean identifying the text as as (sub-) section heading
            'section_reference' : the full reference. Starting at the root node and ending with the value in 'reference'
        '''
        expected_columns = ['indent', 'reference', 'text', 'heading', 'section_reference'] # this is a minimum - it could contain more

        actual_columns = self.document_as_df.columns.to_list()
        for column in expected_columns:
            if column not in actual_columns:
                print(f"{column} not in the DataFrame version of the manual")
                return False
        return True


    # Note: This method will not work correctly if empty values in the dataframe are NaN as is the case when loading
    #       a dataframe form a file without the 'na_filter=False' option. You should ensure that the dataframe does 
    #       not have any NaN value for the text fields. Try running self.document_as_df.isna().any().any() as a test before you get here
    def get_heading(self, section_reference, add_markdown_decorators = False):
        ## NOTE add_markdown_decorators not implemented 
        if not self.reference_checker.is_valid(section_reference):
            return "Not a valid reference"
        remaining_reference = section_reference
        heading = ""
        
        while len(remaining_reference) > 0:
            tmp_df = self.document_as_df.loc[(self.document_as_df['section_reference'] == remaining_reference) & (self.document_as_df['heading'] == True)]
            if len(tmp_df) > 1:
                return f"There was more than one heading for the section reference {section_reference}"
            elif len(tmp_df) == 1:
                if heading == "":
                    heading = tmp_df.iloc[0]["reference"] + " " + tmp_df.iloc[0]["text"] + "."
                else:
                    heading = tmp_df.iloc[0]["reference"] + " " + tmp_df.iloc[0]["text"] + ". " + heading
            remaining_reference = self.reference_checker.get_parent_reference(remaining_reference)

        return heading


    # Note: This method will not work correctly if empty values in the dataframe are NaN as is the case when loading
    #       a dataframe form a file without the 'na_filter=False' option. You should ensure that the dataframe does 
    #       not have any NaN value for the text fields. Try running self.document_as_df.isna().any().any() as a test before you get here
    def get_text(self, section_reference, add_markdown_decorators = True, add_headings = True, section_only = False):

        space = " "
        line_end = "\n"
        heading = ''
        formatted_regulation = ""
        if add_markdown_decorators:
            space = '&nbsp;'
            line_end = "\n\n"
            formatted_regulation = "# "

        # if not self.reference_checker.is_valid(section_reference):
        #     return "The reference did not conform to this documents standard"
        text = ''
        terminal_text_df = self.document_as_df[self.document_as_df['section_reference'].str.startswith(section_reference)]
        if len(terminal_text_df) == 0:
            return f"No section could be found with the reference {section_reference}"
        terminal_text_index = terminal_text_df.index[0]
        terminal_text_indent = 0 # terminal_text_df.iloc[0]['indent']
        for index, row in terminal_text_df.iterrows():
            number_of_spaces = (row['indent'] - terminal_text_indent) * 4
            #set the string "line" to start with the number of spaces
            line = space * number_of_spaces
            if pd.isna(row['reference']) or row['reference'] == '':
                line = line + row['text']
            else:
                if pd.isna(row['text']):
                    line = line + row['reference']
                else:     
                    line = line + row['reference'] + " " + row['text']
            if text != "":
                text = text + line_end
            text = text + line

        if section_reference != '': #i.e. there is a parent
            parent_reference = self.reference_checker.get_parent_reference(section_reference)
            all_conditions = ""
            all_qualifiers = ""
            while parent_reference != "":
                parent_text_df = self.document_as_df[self.document_as_df['section_reference'] == parent_reference]
                conditions = ""
                qualifiers = ""
                for index, row in parent_text_df.iterrows():
                    if index < terminal_text_index:
                        number_of_spaces = (row['indent'] - terminal_text_indent) * 4
                        if conditions != "":
                            conditions = conditions + line_end
                        conditions = conditions + space * number_of_spaces
                        if (row['reference'] == ''):
                            conditions = conditions + row['text']
                        else:
                            conditions = conditions + row['reference'] + " " +  row['text']
                    else:
                        number_of_spaces = (row['indent'] - terminal_text_indent) * 4
                        if (qualifiers != ""):
                            qualifiers = qualifiers + line_end
                        qualifiers = qualifiers + space * number_of_spaces
                        if (row['reference'] == ''):
                            qualifiers = qualifiers + row['text']
                        else:
                            qualifiers = qualifiers + row['reference'] + " " + row['text']

                if conditions != "":
                    all_conditions = conditions + line_end + all_conditions
                if qualifiers != "":
                    all_qualifiers = all_qualifiers + line_end + qualifiers
                parent_reference = self.reference_checker.get_parent_reference(parent_reference)

            if all_conditions != "":
                text = all_conditions +  text
            if all_qualifiers != "":
                text = text + all_qualifiers

        return text

    def get_toc(self):
        toc = StandardTableOfContent(root_node_name = "User Queries", reference_checker = self.reference_checker, regulation_df = self.document_as_df)
        return toc


