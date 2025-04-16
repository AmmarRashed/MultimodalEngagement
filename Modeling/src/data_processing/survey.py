import pandas as pd


def load_participant_answers(user_id, db_path):
    db_con = f"sqlite:///{db_path}"
    survey = pd.read_sql(f"SELECT * FROM answer WHERE user_id='{user_id}'", db_con)
    survey = survey.assign(start_ts=pd.to_datetime(survey.start_ts, format="ISO8601").dt.tz_localize("US/Eastern"),
                           end_ts=pd.to_datetime(survey.end_ts, format="ISO8601").dt.tz_localize("US/Eastern"))
    return survey
