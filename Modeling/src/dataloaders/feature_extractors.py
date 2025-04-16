import numpy as np
import pandas as pd
from scipy import interpolate


def extract_eye_features(df):
    """
    Extract key eye tracking features while maintaining original sampling rate.
    Features: fixations, saccades, pupil diameter, and blink state
    Applies data cleaning using validity flags.

    Args:
        df: DataFrame with eye tracking data

    Returns:
        DataFrame with cleaned features at original sampling rate
    """
    df = df.assign(Timestamp=pd.to_datetime(df.Timestamp, format="mixed"))
    # period_ms = df.Timestamp.diff().dt.total_seconds().mean() * 1000
    period_ms = 1000 / 60
    features = df.copy()

    # Clean fixation data using validity flag
    # fixation_cols = ["FPOGX", "FPOGY", "FPOGD"]
    # features.loc[features["FPOGV"] == 0, fixation_cols] = None
    #
    # # Clean pupil data using validity flags and compute average
    # features.loc[features["LPV"] == 0, "LPD"] = None
    # features.loc[features["RPV"] == 0, "RPD"] = None
    #
    # # Clean saccade data - set to None if fixation is invalid
    # features.loc[features["FPOGV"] == 0, ["SACCADE_MAG", "SACCADE_DIR"]] = None

    # Convert blink IDs to binary blink state
    features["is_blinking"] = (features["BKID"] > 0).astype(int)

    # Select final features while maintaining temporal order
    selected_features = features[
        [
            "FPOGX",
            "FPOGY",
            "FPOGD",  # Fixation position and duration
            "SACCADE_MAG",
            "SACCADE_DIR",  # Saccade metrics
            "LPD", "RPD",  # pupil diameter
            "is_blinking",  # Binary blink state
            "BKDUR",  # Duration of preceding blink
            "BKPMIN",  # Blinks per minute
        ]
    ]
    return selected_features.interpolate()


def extract_eeg_features(df):
    df = df.assign(**{"EQ.OVERALL": df["EQ.OVERALL"].bfill()})
    band_columns = [c for c in df.columns if c.startswith("POW")]
    df = df[["EQ.OVERALL"] + band_columns].dropna(subset=band_columns)
    # df.loc[df["EQ.OVERALL"] < 50, band_columns] = None
    # return df.interpolate().drop("EQ.OVERALL", axis=1).dropna()
    return df.drop("EQ.OVERALL", axis=1)


def extract_hr_features(df):
    df = df.assign(Timestamp=pd.to_datetime(df.Timestamp, format="mixed"))

    # Create continuous time index at 5-second intervals
    full_range = pd.date_range(
        start=df["Timestamp"].min(), end=df["Timestamp"].max(), freq=f"5s"
    )
    # Interpolate missing values
    f = interpolate.interp1d(
        df["Timestamp"].astype(np.int64),
        df["BPM"],
        kind="cubic",
        bounds_error=False,
        fill_value="extrapolate",
    )

    interpolated_bpm = f(full_range.astype(np.int64))

    # Calculate RR intervals in milliseconds
    rr_intervals = (60 / interpolated_bpm) * 1000

    # Create output DataFrame
    rr_df = pd.DataFrame({"Timestamp": full_range, "RR_Interval_ms": rr_intervals})
    return rr_df.drop(["Timestamp"], axis=1)


def calculate_kinematics(df, timestamp_col='timestamp', pose_cols=['pose_Tx', 'pose_Ty', 'pose_Tz']):
    """Calculate velocity and acceleration for pose columns."""
    result = df.copy()

    # Sort by timestamp
    result = result.sort_values(timestamp_col)

    # Calculate time differences in seconds
    dt = result[timestamp_col].diff().mean()

    for col in pose_cols:
        # Velocity
        dp = np.diff(result[col])
        velocity = np.concatenate(([0], dp / dt))
        result[f'velocity_{col}'] = velocity

        # Acceleration
        dv = np.diff(velocity)
        acceleration = np.concatenate(([0], dv / dt))
        result[f'acceleration_{col}'] = acceleration

    return result


def extract_openface_features(df):
    columns = [c for c in df.columns if (c.startswith("AU") and c.endswith("_r")) or c.startswith("pose")]
    df = df[["timestamp", "success"] + columns]
    df = calculate_kinematics(df, pose_cols=["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"])
    return df.drop(["timestamp", "success"], axis=1)
