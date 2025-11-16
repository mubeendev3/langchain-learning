from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_prompt_template = ChatPromptTemplate.from_messages([
    ('system', 'You are an helpful {domain} assistant.'),
    ('human', 'Explain in simple terms, what is {topic}')
])

prompt = chat_prompt_template.invoke({'domain': 'cricket', 'topic': 'runrate'})

print(prompt)