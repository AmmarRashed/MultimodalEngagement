from models import db


class Level(db.Model):
    __tablename__ = "level"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(32), nullable=False, unique=True)

    games = db.relationship("Game", secondary='game_level_association', back_populates='levels')
