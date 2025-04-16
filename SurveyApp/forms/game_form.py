from datetime import datetime

from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField, SelectMultipleField
from wtforms.validators import DataRequired, NumberRange


class GameForm(FlaskForm):
    title = StringField('Title', validators=[DataRequired()])
    year = IntegerField('Year',
                        default=2017,
                        validators=[
                            DataRequired(),
                            NumberRange(min=1990, max=datetime.utcnow().year)
                        ])
    genres = SelectMultipleField("Genre", validators=[DataRequired()], choices=[])
    submit = SubmitField('Add game')
