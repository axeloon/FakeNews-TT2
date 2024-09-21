from pydantic import BaseModel, root_validator
from typing import List, Optional

class UserData(BaseModel):
    user_name: str
    verified: bool
    join_date: str
    followers_info: dict

class Tweet(BaseModel):
    fecha: str
    nombre: str
    usuario: str
    texto: str
    interacciones: str
    imagen: Optional[str]

class SearchUserResponse(BaseModel):
    user_data: UserData
    tweets: List[Tweet]

    @root_validator(pre=True)
    def create_models(cls, values):
        if isinstance(values.get('user_data'), dict):
            values['user_data'] = UserData(**values['user_data'])
        if isinstance(values.get('tweets'), list):
            values['tweets'] = [Tweet(**tweet) if isinstance(tweet, dict) else tweet for tweet in values['tweets']]
        return values