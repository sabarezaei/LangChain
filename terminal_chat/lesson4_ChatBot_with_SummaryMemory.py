from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationSummaryMemory, FileChatMessageHistory   # here we use the ConversationSummaryMemory
#to save the chat history in a .json file, and we can also use it to summarize the chat history
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(verbose=True)     #verbose=True to see the prompt and the response from the model in the console


memory = ConversationSummaryMemory(
    chat_memory=FileChatMessageHistory("messages.json"), # to save all the chat in a .json file
    memory_key="messages",
    return_messages=True,
    llm=chat    # uses the chat model to summarize the chat history
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
    verbose=True             #verbose=True to see the prompt and the response from the model in the console
)

continue_chatting = True

while continue_chatting:
    content = input(">> ")
    if content.lower() in ["exit", "quit"]:
        continue_chatting = False
        print("Exiting chat...")
        

    result = chain({"content": content})

    print(result["text"])
