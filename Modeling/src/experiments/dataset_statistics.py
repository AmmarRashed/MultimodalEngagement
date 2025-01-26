from pathlib import Path
from typing import Dict

import pandas as pd
from joblib import Parallel, delayed
from moviepy.editor import VideoFileClip
from tqdm.notebook import tqdm


def get_filename_from_submission_info(row, extension=".csv"):
    filename = f"{row['participant_id']}_{row['submission_id']}_{row['engagement']}_{row['interest']}_{row['stress']}_{row['excitement']}"
    return filename + extension


class DatasetStats:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.participants_df = pd.read_csv(self.base_path / "Questionnaire/participants.csv")
        self.submissions_df = pd.read_csv(self.base_path / "Questionnaire/submissions.csv")
        self.submissions_df["start_ts"] = pd.to_datetime(self.submissions_df['start_ts'], utc=True).dt.tz_convert(
            "US/Eastern")
        self.submissions_df["end_ts"] = pd.to_datetime(self.submissions_df['end_ts'], utc=True).dt.tz_convert(
            "US/Eastern")

    def get_filepath_from_submission_info(self, row, modality):
        return (self.base_path / "Samples" / str(row["participant_id"]) / modality /
                get_filename_from_submission_info(row, ".mp4" if modality == "OBS" else ".csv"))

    def get_basic_stats(self) -> Dict:
        """Returns basic dataset statistics."""
        stats = {
            'total_participants': len(self.participants_df),
            'total_sessions': len(self.submissions_df),
            'sessions_per_participant': self.submissions_df.groupby('participant_id').size().describe(),
            'sessions_per_game': self.submissions_df['game'].value_counts(),
            'engagement_distribution': self.submissions_df['engagement'].value_counts()
        }

        # Game experience stats
        exp_stats = {
            'fifa_exp': self.participants_df['fifa_exp'].value_counts(),
            'sf_exp': self.participants_df['sf_exp'].value_counts()
        }
        stats.update(exp_stats)

        return stats

    def get_duration_stats(self) -> pd.DataFrame:
        """Calculates session duration statistics."""
        self.submissions_df['duration'] = (
                self.submissions_df['end_ts'] - self.submissions_df['start_ts']).dt.total_seconds()
        return self.submissions_df['duration'].describe()

    def get_obs_stats(self) -> pd.DataFrame:
        """Analyzes OBS recordings statistics using moviepy."""

        def step(row):
            video_path = self.get_filepath_from_submission_info(row, "OBS")
            clip = VideoFileClip(str(video_path))
            duration = clip.duration
            frame_count = int(duration * clip.fps)
            clip.close()
            return (row["participant_id"], row["game"], duration, frame_count, row["engagement"],
                    row["interest"], row["stress"], row["excitement"])

        data = Parallel(n_jobs=-1)(delayed(step)(row) for _, row in tqdm(self.submissions_df.iterrows(), leave=False))
        stats = pd.DataFrame(data,
                             columns=["Participant", "Game",
                                      "Duration", "Frames",
                                      "Engagement", "Interest", "Stress", "Excitement"])

        return stats

    def get_eye_tracking_stats(self) -> pd.DataFrame:
        """Analyzes eye tracking data statistics."""

        def step(row):
            eye_path = self.get_filepath_from_submission_info(row, "EYE")
            if eye_path.exists():
                df = pd.read_csv(eye_path)
                n = len(df)
                n_pogv = len(df["FPOGV"] == 1)
                n_lv = len(df["LPV"] == 1)
                n_rv = len(df["RPV"] == 1)
                return (row["participant_id"], row["game"],
                        n, n_pogv, n_lv, n_rv,
                        row["engagement"], row["interest"], row["stress"], row["excitement"])

        data = Parallel(n_jobs=-1)(delayed(step)(row) for _, row in tqdm(self.submissions_df.iterrows(), leave=False))
        stats = pd.DataFrame(data,
                             columns=["Participant", "Game",
                                      "NumPoints", "ValidFixationPoints", "ValidLeftPoints", "ValidRightPoints",
                                      "Engagement", "Interest", "Stress", "Excitement"])
        return stats

    def get_hr_stats(self) -> pd.DataFrame:
        """Analyzes heart rate data statistics."""

        def step(row):
            hr_path = self.get_filepath_from_submission_info(row, "HR")
            if hr_path.exists():
                df = pd.read_csv(hr_path)
                n = len(df)
                mean_conf = df["Confidence"].mean()
                bpm_var = df["BPM"].var()
                return (row["participant_id"], row["game"], n, mean_conf, bpm_var,
                        row["engagement"], row["interest"], row["stress"], row["excitement"])

        data = Parallel(n_jobs=-1)(delayed(step)(row) for _, row in tqdm(self.submissions_df.iterrows(), leave=False))

        stats = pd.DataFrame(data,
                             columns=["Participant", "Game",
                                      "NumPoints", "MeanConf", "VarBPM",
                                      "Engagement", "Interest", "Stress", "Excitement"])
        return stats

    def get_eeg_stats(self) -> pd.DataFrame:
        """Analyzes EEG data statistics."""

        def step(row):
            eeg_path = self.get_filepath_from_submission_info(row, "EEG")
            if eeg_path.exists():
                df = pd.read_csv(eeg_path)
                mean_quality = df["EQ.OVERALL"].mean()
                std_quality = df["EQ.OVERALL"].std()
                raw_n = len(df)
                band_c = [c for c in df.columns if c.startswith("POW")]
                if len(band_c) > 0:
                    band_n = len(df[band_c[0]].dropna())
                else:
                    band_n = 0
                pm_n = len(df[df["PM.Engagement.IsActive"] == 1])
                return (row["participant_id"], row["game"],
                        mean_quality, std_quality, raw_n, band_n, pm_n,
                        row["engagement"], row["interest"], row["stress"], row["excitement"])

        data = Parallel(n_jobs=-1)(delayed(step)(row) for _, row in tqdm(self.submissions_df.iterrows(), leave=False))

        stats = pd.DataFrame(data,
                             columns=["Participant", "Game",
                                      "MeanQuality", "StdQuality",
                                      "Raw_NumPoints", "Band_NumPoints", "PM_NumPoints",
                                      "Engagement", "Interest", "Stress", "Excitement"])
        return stats

    def get_xbox_stats(self) -> pd.DataFrame:
        """Analyzes EEG data statistics."""

        def step(row):
            xbox = self.get_filepath_from_submission_info(row, "XBOX")
            if xbox.exists():
                df = pd.read_csv(xbox)
                n = len(df)
                n_btns = len(df[df.EventType == "Key"])
                n_analog = len(df[df.EventType == "Absolute"])
                return (row["participant_id"], row["game"],
                        n, n_btns, n_analog,
                        row["engagement"], row["interest"], row["stress"], row["excitement"])

        data = Parallel(n_jobs=-1)(delayed(step)(row) for _, row in tqdm(self.submissions_df.iterrows(), leave=False))

        stats = pd.DataFrame(data,
                             columns=["Participant", "Game",
                                      "NumPoints", "NumBtnEvents", "NumAnalogEvents",
                                      "Engagement", "Interest", "Stress", "Excitement"])
        return stats

    def format_latex_table(self, stats: Dict, caption: str) -> str:
        """Formats statistics into a LaTeX table."""
        latex_str = "\\begin{table}[htb]\n\\centering\n"
        latex_str += f"\\caption{{{caption}}}\n"
        latex_str += "\\begin{tabular}{lr}\n\\toprule\n"

        for key, value in stats.items():
            if isinstance(value, (int, float)):
                latex_str += f"{key} & {value:.2f} \\\\\n"
            elif isinstance(value, pd.Series):
                for idx, val in value.items():
                    latex_str += f"{key}_{idx} & {val:.2f} \\\\\n"

        latex_str += "\\bottomrule\n\\end{tabular}\n\\end{table}"
        return latex_str
