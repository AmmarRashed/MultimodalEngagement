from flask import Blueprint
from controllers.answer_controller import *

answer_bp = Blueprint("answer_bp", __name__)

answer_bp.route("/participate", methods=["GET", "POST"])(participate)
answer_bp.route("/session_no", methods=["GET"])(get_next_session_no)
answer_bp.route("/update_session", methods=["GET"])(update_session)
