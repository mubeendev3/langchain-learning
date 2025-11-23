from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the feedback")

parser2 = PydanticOutputParser[Feedback](pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the followind feedback text into positive, negative \n{feedback} \n{format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

classifier_chain.invoke({"feedback": "I love the product!"})

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n{feedback}",
    input_variables=["feedback"]
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n{feedback}",
    input_variables=["feedback"]
)

conditional_branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "Thank you for your feedback!")
)

chain = classifier_chain | conditional_branch_chain

result = chain.invoke({"feedback": "I hate the product!"})
print(result)

chain.get_graph().print_ascii()