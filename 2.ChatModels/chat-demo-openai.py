from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4", temperature=0, max_tokens=50,)

result = model.invoke("write a poem about pakistan in simple english wording")

print(result.content)