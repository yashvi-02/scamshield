from datetime import datetime
from sqlalchemy import String, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .database import Base

class Message(Base):
    __tablename__="messages"

    id:Mapped[int]=mapped_column(primary_key=True,index=True)
    message_text:Mapped[str]=mapped_column(Text,nullable=False)
    source:Mapped[str]=mapped_column(String(50),default="api")
    created_at:Mapped[datetime]=mapped_column(DateTime,default=datetime.utcnow)

    detection:Mapped["Detection|None"]=relationship(back_populates="message",cascade="all, delete-orphan")

class Detection(Base):
    __tablename__="detections"

    id:Mapped[int]=mapped_column(primary_key=True,index=True)
    message_id:Mapped[int]=mapped_column(ForeignKey("messages.id"),nullable=False)
    verdict:Mapped[str]=mapped_column(String(50),nullable=False)
    risk_level:Mapped[str]=mapped_column(String(30),nullable=False)
    confidence:Mapped[float]=mapped_column(Float,nullable=False)
    scam_type:Mapped[str|None]=mapped_column(String(100),nullable=True)
    recommended_action:Mapped[str|None]=mapped_column(Text,nullable=True)
    created_at:Mapped[datetime]=mapped_column(DateTime,default=datetime.utcnow)

    message:Mapped["Message"]=relationship(back_populates="detection")
    evidence:Mapped[list["Evidence"]]=relationship(back_populates="detection",cascade="all, delete-orphan")

class Evidence(Base):
    __tablename__="evidence"

    id:Mapped[int]=mapped_column(primary_key=True,index=True)
    detection_id:Mapped[int]=mapped_column(ForeignKey("detections.id"),nullable=False)
    evidence_text:Mapped[str]=mapped_column(Text,nullable=False)

    detection:Mapped["Detection"]=relationship(back_populates="evidence")