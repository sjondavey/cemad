from anytree import Node, RenderTree, find, LevelOrderIter, AsciiStyle
import re
import pandas as pd
from cemad_rag.cemad_reference_checker import CEMADReferenceChecker

from regulations_rag.reg_tools import get_regulation_detail
from regulations_rag.embeddings import num_tokens_from_string
        

class TreeNode(Node):
    def __init__(self, name, full_node_name, parent=None, heading_text=''):
        super().__init__(name, parent=parent)
        self.heading_text = heading_text
        self.full_node_name = full_node_name

    # Recursive function to consolidate headings from leaves to root
    def consolidate_from_leaves(self, consolidate_headings):
        """
        Recursively consolidates heading texts from leaf nodes up to the root node.
        
        This method is used to aggregate or summarize information from the bottom of the
        tree (leaf nodes) upwards, allowing for the compilation of heading texts at higher
        levels in the hierarchy based on a user-defined consolidation function.
        
        Parameters:
        - consolidate_headings (callable): A function that takes a list of heading texts from
          child nodes and consolidates them into a single heading text.
        
        Returns:
        - str: The consolidated heading text for this node after processing all child nodes.
        """
        # base case: if the node is a leaf node (no children)
        if not self.children:
            return self.heading_text
        
        # Recursive case: if the node has children
        children_headings = [child.consolidate_from_leaves(consolidate_headings) for child in self.children]
        self.heading_text = consolidate_headings(children_headings)

        return self.heading_text

class Tree:
    def __init__(self, root_id, reference_checker):
        self.root = TreeNode(root_id, "", parent=None, heading_text='')
        self.reference_checker = reference_checker

    def add_to_tree(self, node_str, heading_text=''):
        """
        Adds a new node to the tree or updates an existing node's heading text based on a hierarchical
        node identifier string.

        This method parses the `node_str` using the `reference_checker` to navigate through the tree
        and find the correct position for the new node or to update an existing node.

        Parameters:
        - node_str (str): The hierarchical identifier of the node to add or update.
        - heading_text (str, optional): The heading text for the node. Defaults to an empty string.

        Raises:
        - ValueError: If `node_str` is not a valid node reference according to `reference_checker`.
        """
        if node_str == self.root.name:
            self.root.heading_text = heading_text
            return

        elif not self.reference_checker.is_valid(node_str):
            raise ValueError(f'{node_str} is not a valid node reference')

        node_names = self.reference_checker.split_reference(node_str)

        current_parent = self.root
        full_node_name = ''
        previous_full_node_name = ''  # variable to hold previous full node name

        for i, node_name in enumerate(node_names):
            previous_full_node_name = full_node_name  # update previous node name before adding current node name
            full_node_name = full_node_name + node_name
            found_node = None

            for child in current_parent.children:
                if child.name == node_name:
                    found_node = child
                    break

            # If the node isn't found, create it
            if found_node is None:
                if i == len(node_names) - 1:  # if this is the last node
                    current_parent = TreeNode(node_name, previous_full_node_name + node_name, parent=current_parent, heading_text=heading_text)
                else:
                    current_parent = TreeNode(node_name, previous_full_node_name + node_name, parent=current_parent, heading_text='')
            else:
                current_parent = found_node
            # If this is the last node and it does not have a heading text, assign it
            if i == len(node_names) - 1 and not current_parent.heading_text:
                current_parent.heading_text = heading_text

    def get_node(self, node_str):
        if node_str == self.root.name:
            return self.root
        if not self.reference_checker.is_valid(node_str):
            raise ValueError(f'{node_str} is not a valid node reference')
        # Start search from the root
        current_node = self.root
        node_names = self.reference_checker.split_reference(node_str)
        for node_name in node_names:
            # Look for the node among the children of the current node
            found_node = next((node for node in current_node.children if node.name == node_name), None)
            # If not found, raise a ValueError
            if found_node is None:
                raise ValueError(f"Node with path {node_str} does not exist in the tree")
            # If found, continue searching from this node
            current_node = found_node
        # Return the node we've found
        return current_node

    def print_tree(self):
        for pre, _, node in RenderTree(self.root, style=AsciiStyle()):
            print(f"{pre}{node.name} [{node.heading_text}]")

    # I use this function when extracting the headings from the manual for indexing. There are no tests for it yet!!
    # TODO: Add tests for this
    def _list_node_children(self, node, indent = 0):
        string = ""
        # For each node, check if at least one child has a non-empty heading text
        children_with_text = [child for child in node.children if child.heading_text != '']

        if children_with_text:
            # If any child has non-empty heading text, print all that node's children with their heading text
            for child in node.children:
                if child.parent == self.root:
                    if child.name in self.reference_checker.exclusion_list:
                        string = string + (' ' * indent + f'{child.name}\n')    
                    else:
                        string = string + (' ' * indent + f'{child.name} {child.heading_text}\n')
                else:
                    string = string + (' ' * indent + f'{child.name} {child.heading_text}\n')
                string = string + self._list_node_children(child, indent + 4)
        return string


def build_tree_for_regulation(root_node_name, regs_as_dataframe, reference_checker):
    """
    Constructs a regulation tree from a DataFrame containing regulation entries.
    
    This function builds a tree structure representing the hierarchical relationship of regulations
    starting from a root node. Each regulation or sub-regulation is added as a node in the tree based
    on its 'section_reference'. The tree can be used to navigate through the regulations efficiently.
    
    Parameters:
    - root_node_name (str): The name of the root node of the tree.
    - regs_as_dataframe (pd.DataFrame): DataFrame containing the regulations. Expected to have
      columns 'heading', 'text', and 'section_reference'.
    - reference_checker (object): A Valid_Index object.
      
    Returns:
    - Tree: A tree structure of the regulations.
    
    Raises:
    - ValueError: If any 'section_reference' in the DataFrame is not valid according to `reference_checker`.
    """
    tree = Tree(root_node_name, reference_checker=reference_checker)
    
    for i, row in regs_as_dataframe.iterrows() :
        try:
            heading_text = row['text'] if row['heading'] else ''

            if not reference_checker.is_valid(row['section_reference']):
                raise ValueError(row['section_reference'] + ' is not a valid reference. See row ' + str(i))

            tree.add_to_tree(row['section_reference'], heading_text=heading_text)

        except Exception as e:
            print(f"An error occurred at row {i}:")
            print(regs_as_dataframe.iloc[i])
            print(f"Error message: {e}")
            break

    return tree


def _split_recursive(node, df, token_limit, reference_checker, node_list=[]):
    """    
    Recursively splits nodes based on token limits and collects valid nodes in a list.
    You shouldn't need to call this method. Rather use the "split_tree()" method
    
    Parameters:
    - node (Node): The current node being processed.
    - dataframe (pd.DataFrame): DataFrame containing the regulation details.
    - token_limit (int): The maximum allowed token count per section.
    - reference_checker (callable): Function to check if an index is valid.
    - node_list (list, optional): List to collect nodes meeting the token criteria.
    
    Returns:
    - list: A list of nodes that meet the token criteria.
    """
    if node_list is None:
        node_list = []

    subsection_text = get_regulation_detail(node.full_node_name, df, valid_index_tracker=reference_checker)
    token_count = num_tokens_from_string(subsection_text)

    if token_count > token_limit:
        if not node.children:
            raise Exception(f'Node {node.full_node_name} has no children but has a token count of {token_count} so it cannot be split into nodes that contain fewer tokens that {token_limit}')
        for child in node.children:
            _split_recursive(child, df, token_limit, reference_checker, node_list)
    else:
        node_list.append(node)

    return node_list


def split_tree(node, dataframe, token_limit, reference_checker):
    """
    Splits a tree starting from a given node into sections that don't exceed a token limit.
    
    Initially this is used to set up the base DataFrame using node == root and later it can be used if we want 
    to change the word_limit for a specific piece of regulation to change chunking where it makes sense.

    Parameters:
    - node (Node): The starting node to split the tree.
    - dataframe (pd.DataFrame): DataFrame containing regulation details.
    - token_limit (int): The maximum allowed token count per section.
    - reference_checker (callable): Function to check if an index is valid.
    
    Returns:
    - pd.DataFrame: A DataFrame with columns ['section', 'text', 'token_count'] for each valid node.
    """
    node_list = _split_recursive(node, dataframe, token_limit, reference_checker, node_list=[])
    section_token_count = [[node.full_node_name, 
                            get_regulation_detail(node.full_node_name, dataframe, reference_checker),
                            num_tokens_from_string(get_regulation_detail(node.full_node_name, dataframe, reference_checker))] 
                           for node in node_list]


    return pd.DataFrame(section_token_count, columns=['section_reference', 'text', 'token_count'])


