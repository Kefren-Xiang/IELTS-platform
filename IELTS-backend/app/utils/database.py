import pymysql
from app.config import Config

def query_db(query, params=None):
    connection = pymysql.connect(**Config.DB_CONFIG)
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params or ())
            result = cursor.fetchall()
            return result
    finally:
        connection.close()

def execute_db(query, params=None):
    connection = pymysql.connect(**Config.DB_CONFIG)
    try:
        with connection.cursor() as cursor:
            cursor.execute(query, params or ())
            connection.commit()
    finally:
        connection.close()