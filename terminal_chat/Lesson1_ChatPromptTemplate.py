# the difference between the chat and the completion is that the chat will keep track of the conversation history 
# and use it to generate the response, while the completion will not keep track of the conversation history and will only generate a response
# based on the input.



# here instead of PromptTemplate we use ChatPromptTemplate, which is a special type of prompt template that is designed for chat-based interactions. 
# It allows us to define a sequence of messages that will be used to generate the response. We can use the MessagesPlaceholder to indicate where
# the conversation history should be inserted in the prompt, and we can use the HumanMessagePromptTemplate to define how the user's input should 
# be formatted in the prompt.



from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()  # chatOpenAI because we are going to have a conversation not just a simple completion


prompt = ChatPromptTemplate(
    input_variables=["content"],
    messages=[
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt
)


continue_chatting = True

while continue_chatting:
    content = input(">> ")
    if content.lower() in ["exit", "quit"]:
        continue_chatting = False
        print("Exiting chat...")
        

    result = chain({"content": content})

    print(result["text"])
