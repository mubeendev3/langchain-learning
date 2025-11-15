from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()


model = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=300)

documents = [
	"Capital of Pakistan is Islamabad",
	"Capital of India is New Delhi",
	"Capital of France is Paris",
    "Capital of China is Beijing",
    "Capital of Japan is Tokyo",
]

query = "What is the capital of Pakistan?"

docs_embeddings = model.embed_documents(documents)
query_embedding = model.embed_query(query)

scores = cosine_similarity([query_embedding], docs_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(f"User query: {query}")
print(f"Most similar document: {documents[index]}")
print(f"Similarity score: {score}")

