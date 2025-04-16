from models import db
from models.associations import game_genre_association


class Genre(db.Model):
    __tablename__ = "genre"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(32), nullable=False, unique=True)

    games = db.relationship("Game", secondary=game_genre_association, back_populates='genres')

