from flask import Blueprint
from controllers.user_controller import *

user_bp = Blueprint("user_bp", __name__)

user_bp.route("/register", methods=["GET", "POST"])(register)
user_bp.route('/login', methods=["POST"])(login)
