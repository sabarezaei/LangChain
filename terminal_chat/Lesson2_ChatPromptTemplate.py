# the difference between the chat and the completion is that the chat will keep track of the conversation history 
# and use it to generate the response, while the completion will not keep track of the conversation history and will only generate a response
# based on the input.
# to keep track of the conversation history, we can use the ConversationBufferMemory, which will store the messages in a buffer and return
# them when needed. We set the memory_key to "messages" and return_messages to True so that we can access the messages in the prompt.
# we can also use the FileChatMessageHistory to store the messages in a file, so that we can access them later. 
# This is useful for long-term memory, where we want to keep track of the conversation history even after the program is closed.



from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate , MessagesPlaceholder
from langchain.memory import ConversationSummaryMemory
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()  # chatOpenAI because we are going to have a conversation not just a simple completion


# we defined the memory as ConversationSummaryMemory, which will summarize the conversation history and return it when needed. 
# We set the memory_key to "messages"  so that we can access the messages stored in the memory in the prompt using the messages key word. 
#and return_messages to True so that they messages are stored in form of chat conversation and not just a simple string.
# We also pass the llm to the memory so that it can use it to generate the summary.
memory = ConversationSummaryMemory(memory_key="messages", return_messages=True, llm=chat)


prompt = ChatPromptTemplate(
    input_variables=["content", "messages"],
    messages=[
        MessagesPlaceholder(variable_name="messages"),   # this is where the conversation history will be inserted in the prompt.
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory                       # we wired our memory in here to store the conversation history and use it in the prompt.
)


continue_chatting = True

while continue_chatting:
    content = input(">> ")
    if content.lower() in ["exit", "quit"]:
        continue_chatting = False
        print("Exiting chat...")
        

    result = chain({"content": content})

    print(result["text"])
