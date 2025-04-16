from flask_wtf import FlaskForm
from wtforms import IntegerField, SelectField, EmailField, SubmitField, IntegerRangeField
from wtforms.validators import InputRequired, NumberRange, Email

from forms import EMPTY_CHOICE

EXP_QUESTIONS = {"game1_exp": "fifa.gif", "game2_exp": "street_fighter.gif"}
EXP_CHOICES = ["Beginner", "Amateur", "Semi-Pro", "Pro", "World Class"]


class UserForm(FlaskForm):
    age = IntegerField("",
                       {InputRequired(message="Age is required."),
                        NumberRange(min=14, message="Age must be at least 14")},
                       render_kw={"placeholder": "Enter your age"},
                       id="age")

    sex = SelectField("", [InputRequired(message="Gender is required")],
                      choices=EMPTY_CHOICE + [("Male", "Male"), ("Female", "Female")],
                      name="sex")

    email = EmailField("", [Email(), InputRequired(message="Email is required")],
                       id="email")

    game1_exp = IntegerRangeField("How familiar are you with FIFA games?",
                                  [InputRequired(message="Input Required")],
                                  render_kw={"min": 0, "max": 4, "step": 1, "value": 0},
                                  name="game1_exp")

    game2_exp = IntegerRangeField("How familiar are you with 2.5D fighting games (like StreetFighter)",
                                  [InputRequired(message="InputRequired")],
                                  render_kw={"min": 0, "max": 4, "step": 1, "value": 0},
                                  name="game2_exp")

    submit = SubmitField('Finish Registration')
