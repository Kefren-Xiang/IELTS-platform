import pymysql

class Config:
    DB_CONFIG = {
        "host": "localhost",
        "user": "root",
        "password": "XZH242608xzh",
        "database": "ielts",
        "cursorclass": pymysql.cursors.DictCursor
    }