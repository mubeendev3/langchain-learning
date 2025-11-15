from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
result = model.invoke("write a poem about pakistan in simple english wording")
print(result)