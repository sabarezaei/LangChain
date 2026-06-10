
from langchain_openai import OpenAI
from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import CharacterTextSplitter




text_splitter = CharacterTextSplitter(
    
    separator= "\n",
    chunk_size = 200,
    chunk_overlap =100
    )
# there are variety of data_loader classes available in langchain
# TextLoader, PyPDFLoader, JSONLoader, etc.
# langchain also allows us to point at a remote file (e.g. on Amazon S3) and load it in the same way 
# as a local file. langchain class S3FileLoader can be used to load files from Amazon S3.
# it can load files in a variety of formats, including text, PDF, and JSON.


loader = TextLoader("facts.txt")

docs = loader.load_and_split(text_splitter=text_splitter)

dashed_line = "_".join("_" for _ in range(40))
for doc in docs:
    print(doc.page_content)
    print(dashed_line)
