from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import argparse
from dotenv import load_dotenv

load_dotenv()


parser = argparse.ArgumentParser(description='Generate code using OpenAI API')
parser.add_argument('--language', default='Python', help='Programming language')
parser.add_argument('--task', default='return a list of numbers', help='Task for the function')

args = parser.parse_args()


llm = OpenAI()

code_prompt = PromptTemplate(
    input_variables=["task", "language"],
    template="You are a helpful assistant. Please generate a {language} function that {task}."
)

test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following {language} code: \n {code}"
)

code_chain = LLMChain(llm=llm, prompt=code_prompt , output_key="code")  # renaming the default output key to "code" for clarity
test_chain = LLMChain(llm=llm, prompt=test_prompt , output_key="test")  # renaming the default output key to "test" for clarity

chain = SequentialChain(chains=[code_chain, test_chain], input_variables=["language", "task"], output_variables=["code", "test"])

results = chain({"language": args.language, "task": args.task })

print("Generated Code:")
print(results['code'])

print("\nGenerated Test:")
print(results['test'])  
