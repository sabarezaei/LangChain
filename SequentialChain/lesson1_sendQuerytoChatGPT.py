# simple task #1
# use chatGPT to do a generate a code for a --task in a --language

# ask chat gpt to write a python code that generates numbers 1 to 10 

from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI()

result = llm.invoke("Write a Python function that generates numbers 1 to 10.")

print(result)