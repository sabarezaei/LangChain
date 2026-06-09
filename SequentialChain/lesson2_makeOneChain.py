 # make a chain
 
# receive the output of the first task and give it as input to the model again for the second task 
# such as writing a test for the code




from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv

load_dotenv()

llm = OpenAI()  


code_prompt = PromptTemplate(
    input_variables=["task", "language"],
    template="You are a helpful assistant. Please generate a {language} function that {task}."
)
chain = LLMChain(llm=llm, prompt=code_prompt)

result = chain.invoke({"task": "generates numbers 1 to 10", "language": "Python"}) # we put in a dictionary of variables in here

print(result['text'])