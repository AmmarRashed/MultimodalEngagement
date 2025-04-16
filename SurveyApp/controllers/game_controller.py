from flask import request, render_template, flash, jsonify
from sqlalchemy.exc import IntegrityError
from wtforms import ValidationError

from forms.game_form import GameForm
from forms.user_form import EMPTY_CHOICE
from models import db
from models.Level import Level
from models.game import Game
from models.genre import Genre


def get_game_subfield(value_str, table):
    if not value_str:
        return
    if str.isnumeric(value_str):  # is an id
        value = table.query.get(value_str)
    else:
        value = db.session.query(table).filter_by(name=value_str).first()
    if not value:
        value = table(name=value_str)
        db.session.add(value)
    return value


def add_game_association_field(game, col_name, table, values):
    if not values:
        return
    for str_val in values:
        value = get_game_subfield(str_val, table)
        column = getattr(game, col_name)
        if value not in column:
            column.append(value)
    db.session.commit()


def add_game(title, year, genres=None, levels=None):
    new_game = Game(title=title, year=year)
    add_game_association_field(new_game, col_name="genres", table=Genre, values=genres)
    add_game_association_field(new_game, col_name="levels", table=Level, values=levels)
    try:
        db.session.add(new_game)
        db.session.commit()
        flash("Game inserted successfully.", "success")
    except (IntegrityError, ValidationError) as e:
        db.session.rollback()
        flash(f"Error inserting the game.", "failure")
        print(f"Error inserting game: ({type(e)}): {e}")


def post_game():
    form = GameForm(request.form)
    form.genres.choices = [
        (genre.id, genre.name) for genre in Genre.query.all()
    ]
    # todo
    # form.level.choices = [
    #     (level.id, level.name) for level in Level.query.all()
    # ]
    if request.method == "POST":
        add_game(title=form.title.data, year=form.year.data, genres=form.genres.data
                 # levels=form.levels.data  # todo
                 )

    return render_template("game.html", form=form)


def get_game_levels(game_id, empty_choice=False):
    game = Game.query.get(game_id)
    levels = [(l.id, l.name) for l in game.levels]
    if empty_choice:
        levels = EMPTY_CHOICE + levels
    return {"levels": levels}


def get_selected_game_levels():
    selected_game_id = request.args.get("game_id")
    if selected_game_id:
        return jsonify(get_game_levels(int(selected_game_id), empty_choice=True))
    return jsonify({"levels": EMPTY_CHOICE})


def update_game_levels(game_id, level_str):
    game = Game.query.get(game_id)
    level = get_game_subfield(level_str, Level)
    is_new = False
    if level not in game.levels:
        game.levels.append(level)
        is_new = True
    db.session.commit()
    return level, is_new
