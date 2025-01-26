from flask_wtf import FlaskForm
from wtforms import DateTimeField, SelectField, BooleanField, IntegerRangeField
from wtforms.validators import InputRequired

SURVEY_QUESTIONS = {"engagement": ["Very Bored", "Somewhat Bored", "Neutral", "Somewhat Engaged", "Very Engaged"],
                    "interest": ["Strongly Disliked", "Disliked", "Neutral", "Liked", "Strongly Liked"],
                    "stress": ["Very Relaxed", "Relaxed", "Somewhat Stressed", "Stressed", "Very Stressed"],
                    "excitement": ["Not Excited", "Slightly Excited", "Moderately Excited",
                                   "Extra Excited", "Extremely Excited"]}
RANGE_RENDER_KW = {"min": 0, "max": 4, "step": 1, "value": 0}


class AnswerForm(FlaskForm):
    game = SelectField("Game:", [InputRequired(message="Game is required")],
                       choices=[], coerce=lambda x: int(x) if x else x,
                       name="game")

    game_level = SelectField("Level:", [InputRequired(message="Level is required")],
                             name="game_level")

    is_trial = BooleanField("Trial:", name="trial_cb")

    engagement = IntegerRangeField("How engaged did you feel in the last game session?",
                                   [InputRequired()],
                                   render_kw=RANGE_RENDER_KW,
                                   name="engagement")

    interest = IntegerRangeField("How much did you enjoy the last game session?", [InputRequired()],
                                 render_kw=RANGE_RENDER_KW,
                                 name="interest")

    stress = IntegerRangeField("How stressed did you feel in the last game session?", [InputRequired()],
                               render_kw=RANGE_RENDER_KW,
                               name="stress")

    excitement = IntegerRangeField("How excited did you feel in the last game session?", [InputRequired()],
                                   render_kw=RANGE_RENDER_KW,
                                   name="excitement")

    start_ts = DateTimeField("Started at:", [InputRequired()], name="start_ts", render_kw={"readonly": True})
    end_ts = DateTimeField("Ended at:", [InputRequired()], name="end_ts", render_kw={"readonly": True})
