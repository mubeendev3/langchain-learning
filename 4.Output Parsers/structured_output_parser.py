from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

class Facts(BaseModel):
    fact_1: str = Field(description="The first fact about the topic")
    fact_2: str = Field(description="The second fact about the topic")
    fact_3: str = Field(description="The third fact about the topic")

parser = PydanticOutputParser(pydantic_object=Facts)

template = PromptTemplate(
    template="Give me 3 facts about {topic} \n{format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template.format(topic="AI")
result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)