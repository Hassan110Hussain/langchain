import email
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class Student(BaseModel):
    name: str = "nitish"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=5)


new_stu = {"age": "32", "email": "abcd@gmail.com"}

stu = Student(**new_stu)

stu_dict = dict(stu)

print(stu_dict["age"])

stu_json = stu.model_dump_json()
