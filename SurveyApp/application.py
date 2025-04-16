from flask import Flask, render_template, redirect, session
from flask_migrate import Migrate

from controllers.game_controller import add_game
from controllers.user_controller import add_user
from flask_session import Session
# noinspection PyUnresolvedReferences
from models import user, game, answer, genre, db
from routes.answer_routes import answer_bp
from routes.game_routes import game_bp
from routes.user_routes import user_bp

application = Flask(__name__)
application.config.from_object('config')

db.init_app(application)
migrate = Migrate(application, db)
with application.app_context():
    db.create_all()

application.register_blueprint(game_bp, url_prefix="/game/")
application.register_blueprint(answer_bp, url_prefix="/answer/")
application.register_blueprint(user_bp, url_prefix="/")

Session(application)


@application.route('/_init', methods=["GET"])
def init_experiment():
    # adding main games
    
    add_game("FIFA23", 2023, ["Sports"],
             ["Beginner", "Amateur", "Semi-Pro", "Professional", "World Class", "Legendary"])
    add_game("Street Fighter V", 2015, ["2.5D Fighting"], [f"({i})" for i in range(1, 9)])
    
    # adding test user
    add_user(age=0, sex="A", email="admin", id=111, game1_exp=3, game2_exp=3)
    return redirect("/")


@application.route('/', methods=["GET"])
def home():
    session.clear()
    return render_template("home.html")


if __name__ == '__main__':
    application.run()
