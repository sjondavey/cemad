import shutil
import os



def deploy(base_folder):
    '''
    base_folder = ".." when running from "working" folder

    ''' 
    destination = "d:/code/chat/cemad_rag"

    items_to_copy = [base_folder + "/cemad_rag/",
                     base_folder + "/.gitignore",
                     base_folder + "/app.py",
                     base_folder + "/logging_config.py",
                     base_folder + "/footer.py",
                     base_folder + "/publication_icon.jpg",
                     base_folder + "/streamlit_common.py",
                     base_folder + "/streamlit_pages/",
                     base_folder + "/.streamlit/secrets.toml",
                     base_folder + "/.streamlit/config.toml",
                     base_folder + "/inputs/",
                     base_folder + "/.env/",
                     ]



    for item_path in items_to_copy:
        # Ensure the item_path is normalized
        item_path = os.path.normpath(item_path)
        
        # Create new destination path
        relative_path = os.path.relpath(item_path, base_folder)
        new_destination = os.path.join(destination, relative_path)
        
        if os.path.isfile(item_path):
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(new_destination), exist_ok=True)
            
            # Copy the file
            shutil.copy(item_path, new_destination)
        elif os.path.isdir(item_path):
            # Ensure the destination directory does not already exist
            if os.path.exists(new_destination):
                shutil.rmtree(new_destination)

            # Copy the entire directory and its contents
            shutil.copytree(item_path, new_destination)



