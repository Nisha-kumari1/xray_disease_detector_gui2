from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    email: str
    full_name: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class PredictionBase(BaseModel):
    disease_probabilities: str
    heatmap_image_path: Optional[str] = None
    bounding_box_image_path: Optional[str] = None
    report_text: Optional[str] = None

class Prediction(PredictionBase):
    id: int
    scan_id: int

    class Config:
        from_attributes = True

class ScanBase(BaseModel):
    original_image_path: str

class Scan(ScanBase):
    id: int
    upload_time: datetime
    owner_id: int
    prediction: Optional[Prediction] = None

    class Config:
        from_attributes = True
