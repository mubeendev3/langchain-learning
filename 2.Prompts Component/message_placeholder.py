from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate


chat_history = []

# Chat Template with Message Placeholder
chat_template = ChatPromptTemplate(
    [
        ('system', 'You are an helpful customer support agent.'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{input}')
    ]
)

# Load chat history from file
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

# Create Prompt Template
final_prompt = chat_template.invoke({
    'chat_history': chat_history,
    'input': 'I have not received my refund yet.'
})

print(final_prompt)