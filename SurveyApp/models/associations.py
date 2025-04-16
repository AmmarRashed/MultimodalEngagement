from models import db

game_genre_association = db.Table('game_genre_association',
                                  db.Column('game_id', db.Integer, db.ForeignKey('game.id'), primary_key=True),
                                  db.Column('genre_id', db.Integer, db.ForeignKey('genre.id'), primary_key=True))


class GameLevelAssociation(db.Model):
    __tablename__ = "game_level_association"

    game_id = db.Column('game_id', db.Integer, db.ForeignKey('game.id'), primary_key=True)
    level_id = db.Column('level_id', db.Integer, db.ForeignKey('level.id'), primary_key=True)
