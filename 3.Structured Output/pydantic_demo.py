from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = "Mubeen"
    age: Optional[int] = None
    email: EmailStr = "mubeen@gmail.com"
    cgpa: float = Field(ge=0, le=4, description="A decimal value representing the student's CGPA and it must be between 0 and 4")

new_student = {"age": "22", "cgpa": 3.64}

student = Student(**new_student)

student_dict = student.model_dump()
print(student_dict['name'])