
from langchain_openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
#there are variety of different types of embeddings that we can use
from langchain_openai import OpenAIEmbeddings

from langchain_chroma import Chroma

load_dotenv()

embeddings = OpenAIEmbeddings()


text_splitter = CharacterTextSplitter(
    
    separator= "\n",
    chunk_size = 200,
    chunk_overlap =100
    )


loader = TextLoader("facts.txt")

docs = loader.load_and_split(text_splitter=text_splitter)

db = Chroma.from_documents(   #from_documents is to save the data into emb
    docs,                     # we are telling Chroma to calculate the vectore database for this documents 
    embedding=embeddings,
    persist_directory = "emb"  # the embeddings are going to be saved in the emb SQLite database inside my directory
)


# we calculate the embeddings once and save them and later use them, so we dont calculate them multiple times

dashed_line = "_".join("_" for _ in range(40))
for doc in docs:
    print(doc.page_content)
    print(dashed_line)


results = db.similarity_search_with_score(
    "what is an interesting fact about english language?",  # this is our question that we want to find the answer for, inside our documents
    k = 3   # this defines the number of chunks of data you want your algorithm return to you
    )

for result in results: 
    print(result[1])  # this is the similarity score between the question and the chunk of the document that is returned to us
    print(result[0].page_content)
    print(dashed_line)

    print(result)