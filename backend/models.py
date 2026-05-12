from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, Float
from sqlalchemy.orm import relationship
from datetime import datetime
import database

class User(database.Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    full_name = Column(String)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)

    scans = relationship("Scan", back_populates="owner")

class Scan(database.Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True)
    original_image_path = Column(String) # Path to the uploaded original file
    upload_time = Column(DateTime, default=datetime.utcnow)
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    owner = relationship("User", back_populates="scans")
    prediction = relationship("Prediction", back_populates="scan", uselist=False)

class Prediction(database.Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    scan_id = Column(Integer, ForeignKey("scans.id"))
    
    # Store results as a JSON string for flexibility with multi-disease
    disease_probabilities = Column(String) 
    
    heatmap_image_path = Column(String, nullable=True) # XAI heatmap
    bounding_box_image_path = Column(String, nullable=True) # Localization
    
    report_text = Column(String, nullable=True) # Generated radiologist report
    
    scan = relationship("Scan", back_populates="prediction")
