from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person", ge=18)
    city: str = Field(description="The city of the person belongs to")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n{format_instructions}',
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Without Using Chain
# prompt = template.format(place="Pakistan")
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(prompt)
# print(final_result)

# Using Chain
chain = template | model | parser
result = chain.invoke({"place": "Indonesia"})
print(result)