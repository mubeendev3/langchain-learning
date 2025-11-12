from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# Use a model that supports chat completion through Hugging Face Inference API
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of Pakistan?")
print(result.content)