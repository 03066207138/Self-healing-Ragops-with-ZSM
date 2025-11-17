# app/api/db.py
from sqlalchemy import create_engine, Column, Integer, String, Float, LargeBinary, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# -------------------- Database Setup --------------------
DB_PATH = "sqlite:///app/data/ragops.db"
engine = create_engine(DB_PATH, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# -------------------- Models --------------------
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    embedding = Column(LargeBinary, nullable=True)  # store numpy embeddings as bytes
    created_at = Column(DateTime, default=datetime.utcnow)


class Metric(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, nullable=False)
    latency_ms = Column(Float)
    coverage = Column(Float)
    faithfulness = Column(Float)
    cost = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

# -------------------- Create tables --------------------
def init_db():
    Base.metadata.create_all(bind=engine)
