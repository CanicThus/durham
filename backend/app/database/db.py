import psycopg2
from sqlmodel import Session, create_engine, select
from app.core.config import settings

# connect to db
engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI))

print("connect to postgres successfully")





