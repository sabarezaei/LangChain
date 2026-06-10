
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_community.document_loaders import TextLoader



#load_dotenv()

# there are variety of data_loader classes available in langchain
# TextLoader, PyPDFLoader, JSONLoader, etc.
# langchain also allows us to point at a remote file (e.g. on Amazon S3) and load it in the same way 
# as a local file. langchain class S3FileLoader can be used to load files from Amazon S3.
# it can load files in a variety of formats, including text, PDF, and JSON.


loader = TextLoader("facts.txt")

docs = loader.load()


print(docs)