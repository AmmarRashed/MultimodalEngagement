from flask import render_template, request, session, flash, jsonify, redirect
from sqlalchemy.exc import IntegrityError
from wtforms import ValidationError

from controllers.game_controller import get_game_levels, update_game_levels
from forms.answer_form import AnswerForm, SURVEY_QUESTIONS
from forms.user_form import EMPTY_CHOICE
from models import db
from models.answer import Answer
from models.game import Game


def get_next_session_no():
    result = _get_next_session_no(
        user_id=request.args.get("user_id"),
        game_id=request.args.get("game_id"),
        level_id=request.args.get("game_level")
    )
    return jsonify(result)


def _get_next_session_no(user_id, game_id, level_id):
    result = db.session.query(Answer).filter(
        Answer.user_id == user_id, Answer.game_id == game_id, Answer.level_id == level_id
    ).count()
    return {"session_no": 1 if result is None else result + 1}


def post_answer(data):
    kwargs = {k: data.get(k) for k in list(SURVEY_QUESTIONS.keys()) + ["start_ts", "end_ts"]}
    new_answer = Answer(
        user_id=session.get("user_id"),
        game_id=data.get("game"),
        level_id=data.get('game_level'),
        session_no=session.get("session_no"),
        **kwargs)

    try:
        db.session.add(new_answer)
        db.session.commit()
        flash(f"Recorded answer successfully.", "success")
        return True
    except (IntegrityError, ValidationError) as e:
        db.session.rollback()
        flash(f"Error recording the answer.", "failure")
        print(f"Error recording answer: ({type(e)}): {e}")


def update_session():
    game_id = request.args.get("game_id")
    session["game_id"] = game_id

    game_level = request.args.get("game_level")
    session["game_level"] = game_level

    session_no = _get_next_session_no(session["user_id"], game_id, game_level)["session_no"]
    session["session_no"] = session_no

    return jsonify({"session_no": session_no})


def clear_answers(form, increment_session=True):
    for q in list(SURVEY_QUESTIONS.keys()) + ["start_ts", "end_ts"]:
        form[q].data = None
    form.start_ts.raw_data = None
    form.end_ts.raw_data = None
    if increment_session:
        session["session_no"] += 1


def participate():
    if session.get("user_id") is None:
        return redirect("/")
    form = AnswerForm(request.form)
    is_trial = form.is_trial.data
    form.game.choices = EMPTY_CHOICE + [(g.id, g.title) for g in Game.query.all()]
    if session.get("game_id"):
        form.game_level.choices = get_game_levels(game_id=session.get("game_id"), empty_choice=True)["levels"]
    if is_trial:
        flash(f"Recorded answer successfully.", "success")
        clear_answers(form, increment_session=False)
    elif request.method == "POST":
        level, is_new = update_game_levels(session.get("game_id"), form.data["game_level"])
        if is_new:
            form.game_level.choices.append((level.id, level.name))
        session["game_level"] = level.id
        form.game_level.data = str(level.id)
        if post_answer(form.data):
            clear_answers(form)
    return render_template("survey.html", form=form, session=session,
                           questions=SURVEY_QUESTIONS, enumerate=enumerate)
