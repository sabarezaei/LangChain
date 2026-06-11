from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings() 
chat = ChatOpenAI() 
db = Chroma( # we want an instance of Chroma to do some similarity search but we dont want to add to the database 
            # we pass in the "emb" database that we prepared in the previouse stage 
            persist_directory = "emb", 
            embedding_function=embeddings 
            # this is the function that converts the input text to embedings
            ) 
            
            
retriever = db.as_retriever()

chain = RetrievalQA.from_chain_type( llm = chat, retriever = retriever, chain_type = "stuff" ) 


result = chain.invoke("what is an interesting fact about the english language?") 
print(result)