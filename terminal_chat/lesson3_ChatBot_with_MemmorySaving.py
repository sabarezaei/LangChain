from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(verbose=True)

memory = ConversationBufferMemory(
    chat_memory=FileChatMessageHistory("messages.json"), # to save all the chat in a .json file
    memory_key="messages",
    return_messages=True
)


prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
    verbose=True
)

continue_chatting = True

while continue_chatting:
    content = input(">> ")
    if content.lower() in ["exit", "quit"]:
        continue_chatting = False
        print("Exiting chat...")
        

    result = chain({"content": content})

    print(result["text"])
