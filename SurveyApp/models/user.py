from sqlalchemy.orm import relationship

from models import db


class User(db.Model):
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer, nullable=False)
    sex = db.Column(db.String(1), nullable=False)
    email = db.Column(db.String(128), nullable=False, unique=True)

    game1_exp = db.Column(db.Integer)
    game2_exp = db.Column(db.Integer)

    answer = relationship("Answer", uselist=False, backref="user")
