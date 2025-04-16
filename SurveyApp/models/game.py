from models import db
from models.associations import game_genre_association


class Game(db.Model):
    __tablename__ = "game"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    title = db.Column(db.String, unique=True)
    year = db.Column(db.Integer, nullable=True)

    genres = db.relationship("Genre", secondary=game_genre_association, back_populates='games')
    levels = db.relationship("Level", secondary="game_level_association", back_populates='games')
