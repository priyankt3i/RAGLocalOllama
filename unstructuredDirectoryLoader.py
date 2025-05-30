import glob
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from pathlib import Path

class CustomDirectoryLoader:
    def __init__(self, directory_path: str):
        """
        Initialize the loader with a directory path and a glob pattern.
        :param directory_path: Path to the directory containing files to load.
        """
        self.directory_path = directory_path

    def load(self) -> List[Document]:
        """
        Load all files matching the glob pattern in the directory using UnstructuredFileLoader.
        :return: List of Document objects loaded from the files.
        """
        #os.chdir(str(self.directory_path))
        documents = []
        # Construct the full glob pattern
        # Iterate over all files matched by the glob pattern
        files = (p.resolve() for p in Path(str(self.directory_path)).glob("**/*") if p.suffix in {".txt", ".csv", ".pdf", ".msg"})
        for file_path in files:
            print(file_path)
            # Use UnstructuredFileLoader to load each file
            loader = UnstructuredFileLoader(file_path=file_path, mode="elements")
            try:
                docs = loader.load()
                documents.extend(docs)
                #print(str(docs))
            except Exception:
                print('Failed to load : ' + str(file_path))
                pass
        return documents
    
def main():
    #directory_loader = CustomDirectoryLoader(directory_path='./data/samples')
    directory_loader = CustomDirectoryLoader(directory_path='C:\\Users\\kpriyank\\VSCodeRepo\\langchain-rag-main\\examples')
    docs = directory_loader.load()
    print(docs)

if __name__ == "__main__":
    main()












