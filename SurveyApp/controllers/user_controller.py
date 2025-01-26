import hashlib
import random

from flask import request, render_template, flash, redirect, url_for, session
from sqlalchemy.exc import IntegrityError
from wtforms import ValidationError

from forms.user_form import UserForm, EXP_QUESTIONS, EXP_CHOICES
from models import db
from models.user import User


def generate_user_id(age, sex, email):
    input_string = f"{age}{sex}{email}"
    hash_value = hashlib.sha256(input_string.encode()).hexdigest()
    id_number = int(hash_value[:3], 16) % 1000
    if id_number < 100:  # Ensure it's at least 3 digits long
        id_number += 100 * random.randint(1, 9)
    return id_number


def check_id_exists(id):
    return User.query.get(id) is not None


def add_user(**kwargs):
    sex = kwargs.get("sex")
    email = kwargs.get("email")
    assert len(sex) == 1, "sex should be a one-character string"
    assert len(email) > 0, "email should not be an empty string"
    id = kwargs.get("id")
    age = kwargs.get("age")
    if id is None:
        id = generate_user_id(age, sex, email)
        while check_id_exists(id):
            id = random.randint(100, 999)
    kwargs["id"] = id
    new_user = User(
        **kwargs
    )
    try:
        db.session.add(new_user)
        db.session.commit()
        return new_user
    except (IntegrityError, ValidationError) as e:
        db.session.rollback()
        flash(f"Error registering the user.", "failure")
        print(f"Error registering user: ({type(e)}): {e}")


def register():
    form = UserForm(request.form)
    if request.method == "POST":
        age = form.age.data
        sex = form.sex.data[0]
        email = form.email.data
        game1_exp = form.game1_exp.data
        game2_exp = form.game2_exp.data
        new_user = add_user(age=age, sex=sex, email=email,
                            game1_exp=game1_exp, game2_exp=game2_exp)
        if new_user:
            session["user_id"] = new_user.id
            return redirect(url_for("answer_bp.participate"))
    return render_template("user.html", form=form, questions=EXP_QUESTIONS, choices=EXP_CHOICES,
                           enumerate=enumerate)


def login():
    email_or_id = request.form.get("email").lower().strip()
    if str.isnumeric(email_or_id):
        user = User.query.get(email_or_id)
    else:
        user = db.session.query(User).filter_by(email=email_or_id).first()

    if user:
        session["user_id"] = user.id
        return redirect(url_for("answer_bp.participate"))
    flash("Invalid Email or User ID", "failure home")
    return render_template("home.html")
