from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

documents = [
	"Capital of Pakistan is Islamabad",
	"Capital of India is New Delhi",
	"Capital of France is Paris"
]

result = model.embed_documents(documents)

print(result)