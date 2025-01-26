from controllers.game_controller import *
from flask import Blueprint

game_bp = Blueprint("game_bp", __name__)

game_bp.route("/", methods=["GET", "POST"])(post_game)
game_bp.route("/levels", methods=["GET"])(get_selected_game_levels)
