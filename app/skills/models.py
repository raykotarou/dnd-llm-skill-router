from pydantic import BaseModel


class Skill(BaseModel):
    id: str
    name: str
    description: str
    content: str
    path: str
