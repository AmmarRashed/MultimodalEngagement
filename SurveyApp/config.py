import os

SECRET_KEY = os.urandom(32)

basedir = os.path.abspath(os.path.dirname(__file__))

DEBUG = True

DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")
SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_PATH}"
SQLALCHEMY_TRACK_MODIFICATIONS = False

SESSION_PERMANENT = False
SESSION_TYPE = "filesystem"
