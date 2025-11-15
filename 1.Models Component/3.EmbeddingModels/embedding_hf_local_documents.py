from langchain_huggingface import HuggingFaceEmbeddings

documents = [ "Capital of Pakistan is Islamabad", "Capital of India is New Delhi", "Capital of France is Paris" ]

model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

result = model.embed_documents(documents)

print(result)