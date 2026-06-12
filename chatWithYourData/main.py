from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Embeddings
embeddings = OpenAIEmbeddings()

# Vector database
db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)

# Retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# Prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Answer the user's question using only the provided context.

Context:
{context}
"""
    ),
    ("human", "{input}")
])

# Combine retrieved docs
document_chain = create_stuff_documents_chain(
    llm,
    prompt
)

# Create retrieval chain
chain = create_retrieval_chain(
    retriever,
    document_chain
)

# Ask question
response = chain.invoke({
    "input": "What is an interesting fact about the English language?"
})

print(response["answer"])