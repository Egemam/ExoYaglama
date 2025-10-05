from multiprocessing import connection

from fastapi import FastAPI, Path
import psycopg2
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from urlparse import urlparse
import os
from dotenv import load_dotenv

app = FastAPI()

origins = ["*"]

load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def connect():
    URI = os.getenv('URI')
    result = urlparse(URI)
    connection = psycopg2.connect(
        database=result.path[1:],
        user=result.username,
        password=result.password,
        host=result.hostname,
        port=result.port,
        sslmode='require'
    )
    cursor = connection.cursor()
    connection.autocommit = True
    print("Database connection established.")
    return [cursor, connection]

def disconnect(cursor):
    cursor.close()

class Input(BaseModel):
    test: str

@app.get("/fetch-data")
async def fetch_data():
    connection, cursor = connect()
    result = cursor
    disconnect(cursor)
    return result