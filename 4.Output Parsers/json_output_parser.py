from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

parser = JsonOutputParser()

model = ChatOpenAI(model="gpt-4o-mini")

template = PromptTemplate(
    template='Give me 5 facts about {topic} \n{format_instructions}',
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Way 1 without using chain
# prompt = template.format()
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

# Way 2 using chain
chain = template | model | parser
result = chain.invoke({"topic": "AI"})
print(result)