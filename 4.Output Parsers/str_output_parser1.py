from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


model = ChatOpenAI(model="gpt-4o-mini")

# 1st Prompt
template1 = PromptTemplate(
    template= "Write a detailed repot on {topic}",    
    input_variables=["topic"]
)

# 2nd Prompt
template2 = PromptTemplate(
    template= "Write a 5 lines summary of the following text {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "AI"})

print(result)