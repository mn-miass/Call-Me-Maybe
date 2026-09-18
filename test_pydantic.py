from pydantic import BaseModel


class student(BaseModel):
    name: str
    age: int = 18
    subject: str


student_a = student(name="12", age="12.5", subject="test")

print(student_a)