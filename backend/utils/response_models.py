from datetime import date, datetime
from pydantic import BaseModel, root_validator, Field
from typing import List, Optional

class UserData(BaseModel):
    user_name: str
    verified: bool
    join_date: Optional[str]
    followers_info: dict

class Tweet(BaseModel):
    fecha: str
    nombre: str
    usuario: str
    texto: str
    interacciones: str
    imagen: Optional[str]
    is_false: bool = False

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

class NoticiaBase(BaseModel):
    source: str
    title: str
    content: str
    publication_date: date
    author: str

class NoticiaBaseResponse(NoticiaBase):
    pass

class NoticiaResponse(NoticiaBase):
    id: int
    created_at: datetime
    updated_at: datetime

class EmolNoticiaResponse(BaseModel):
    id: str
    url: str
    title: str
    subtitle: str
    content: str
    publication_date: str
    publication_time: str
    category: str
    author: str
    publisher: str = Field(default="EMOL")
    headline: str = Field(default="historic")

class EmolHistoricoResponse(BaseModel):
    noticias: List[EmolNoticiaResponse]
    total: int

class CsvRequest(BaseModel):
    csv_path: str

class TrainingResponseModel(BaseModel):
    name_model: str
    status: str
    accuracy_train: float | None = None
    accuracy: float | None = None
    precision: float | None = None
    precision_train: float | None = None
    recall: float | None = None
    recall_train: float | None = None
    f1_score: float | None = None
    f1_score_train: float | None = None
    message: str | None = None
    feature_importances: List[float] | None = None
    feature_names: List[str] | None = None
    y_true: List[int] | None = None
    y_pred: List[int] | None = None
    y_prob: List[float] | None = None