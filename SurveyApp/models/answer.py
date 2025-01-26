from sqlalchemy import ForeignKey, UniqueConstraint

from models import db


class Answer(db.Model):
    __tablename__ = "answer"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, ForeignKey('user.id'))
    game_id = db.Column(db.Integer, ForeignKey('game.id'))
    level_id = db.Column(db.Integer, ForeignKey('level.id'))
    session_no = db.Column(db.Integer, nullable=False)
    start_ts = db.Column(db.DateTime, nullable=False)
    end_ts = db.Column(db.DateTime, nullable=False)
    engagement = db.Column(db.Integer, nullable=False)
    interest = db.Column(db.Integer, nullable=False)
    stress = db.Column(db.Integer, nullable=False)
    excitement = db.Column(db.Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint('user_id', 'game_id', 'level_id', 'session_no', name='_unique_user_game_session'),
    )
