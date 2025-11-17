from langchain_openai import ChatOpenAI
from typing import Annotated, Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

# Schema Definition
json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "Write down all key themes discussed"
        },
        "summary": {
            "type": "string",
            "description": "A summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": [
                "Pos",
                "Neg",
                "Neut"
            ],
            "description": "The sentiment of the review"
        },
        "pros": {
            "type": [
                "array",
                "null"
            ],
            "items": {
                "type": "string"
            },
            "description": "Return ONLY if user explicitly lists pros. Do NOT infer pros. If user does NOT explicitly list pros, return an empty list."
        },
        "cons": {
            "type": [
                "array",
                "null"
            ],
            "items": {
                "type": "string"
            },
            "description": "Return ONLY if user explicitly lists cons. Do NOT infer cons. If user does NOT explicitly list cons, return an empty list."
        }
    },
    "required": [
        "key_themes",
        "summary",
        "sentiment"
    ]
}

# class Review(BaseModel):

#     key_themes: list[str] = Field(description="Write down all key themes discussed")
#     summary: str = Field(description="A summary of the review")
#     sentiment: Literal["Pos", "Neg", "Neut"] = Field(description="The sentiment of the review")
#     pros: Optional[list[str]] = Field(default = None, description="Return ONLY if user explicitly lists pros. Do NOT infer pros. If user does NOT explicitly list pros, return an empty list.")
#     cons: Optional[list[str]] = Field(default = None, description="Return ONLY if user explicitly lists cons. Do NOT infer cons. If user does NOT explicitly list cons, return an empty list.")


structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it's an absolute powerhouse! 
The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I'm gaming, multitasking, 
or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast 
charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. 
What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images 
even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x 
loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung's One UI still 
comes with bloatware—why do I need five different Samsung apps for things Google already provides? 
The $1,300 price tag is also a hard pill to swallow.

Pros:
- Insanely powerful processor (great for gaming and productivity)
- Stunning 200MP camera with incredible zoom capabilities
- Long battery life with fast charging
- S-Pen support is unique and useful
""")


# print(result)
print(result)
print(result["summary"])
print(result["sentiment"])
print(result["key_themes"])
print(result["pros"])
print(result["cons"])
