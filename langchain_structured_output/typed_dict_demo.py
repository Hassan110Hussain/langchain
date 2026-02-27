from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

new_person: Person = {'name':'Hassan','age':23} #doesnt validate at runtime even if the datatypes are incorrect

print(new_person)