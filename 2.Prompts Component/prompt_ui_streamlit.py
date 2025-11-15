# Import required libraries
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate

# Load environment variables (like API keys) from .env file
load_dotenv()

# Initialize the OpenAI chat model
model = ChatOpenAI(model="gpt-4o-mini")

# Set up the Streamlit UI
st.title("LangChain Prompt UI")
st.header("Research Tool")

# User input: Select research paper to summarize
paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

# User input: Select the style of explanation
style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

# User input: Select the length of explanation
length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)

# Create a prompt template with placeholders for user inputs
template = PromptTemplate(
    template=""" 
    Please summarize the research paper titled "{paper_input}" with the following specifications:

    Explanation Style: {style_input}
    Explanation Length: {length_input}

    Mathematical Details:
    - Include relevant mathematical equations if present in the paper.
    - Explain the mathematical concepts using simple, intuitive code snippets where applicable.

    Analogies:
    - Use relatable analogies to simplify complex ideas.

    If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.

    Ensure the summary is clear, accurate, and aligned with the provided style and length.
    """,
    input_variables=["paper_input", "style_input", "length_input"]
)

# Fill in the template with user's selected values
prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

# When user clicks Submit button, generate the summary
if st.button("Submit"):
    # Call the AI model with the formatted prompt
    result = model.invoke(prompt)
    # Display the generated summary
    st.write(result.content)