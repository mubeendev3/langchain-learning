from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model1 = ChatOpenAI(model="gpt-4o-mini")
model2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n{text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n{text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the following notes and question answers into a single document \n{notes} \n{quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

parallel_chain = RunnableParallel[dict](
    notes = prompt1 | model1 | parser,
    quiz = prompt2 | model2 | parser
)

merge_chain = prompt3 | model1 | parser

final_chain = parallel_chain | merge_chain


text = """
Artificial Intelligence (AI) is a multidisciplinary field of computer science focused on creating systems capable of performing tasks that traditionally require human intelligence. These tasks include, but are not limited to, understanding natural language, recognizing patterns, solving problems, learning from experience, and making decisions.

The history of AI dates back to the mid-20th century, with early computational theories and the invention of programmable digital computers. Early AI research was ambitious, seeking to build machines that mimicked all aspects of human cognition. Over time, researchers realized the complexity of human intelligence, leading to more focused subfields like machine learning, computer vision, and natural language processing.

Machine learning, a subset of AI, enables computer systems to improve their performance on tasks through experience, without being explicitly programmed for each scenario. Deep learning, inspired by the structure and function of the human brain, uses artificial neural networks with many layers to process data and extract high-level features.

AI applications are widespread and growing rapidly, transforming various sectors such as healthcare, finance, transportation, entertainment, and manufacturing. In healthcare, AI assists in diagnosing diseases, analyzing medical images, and personalizing treatment plans. In finance, AI models can detect fraudulent transactions and predict market trends. Self-driving cars, voice assistants, and recommendation systems are other prominent examples of AI integration into daily life.

Despite its advancements, AI faces challenges, including ethical considerations, transparency, and the need for large quantities of quality data. Concerns about job displacement, bias in AI models, and decision accountability continue to prompt philosophical and regulatory discussions worldwide.

Looking forward, ongoing AI research aims to enhance system robustness, interpretability, and fairness. As AI evolves, it promises to become an even more integral part of society, augmenting human capabilities and driving innovation across countless domains.
"""

final_result = final_chain.invoke({"text": text})
print(final_result)

final_chain.get_graph().print_ascii()
