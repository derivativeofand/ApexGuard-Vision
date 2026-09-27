"""
==========================================================================
F1 SAFETY CAR PREDICTION - FULL PIPELINE
==========================================================================

GOAL: Predict whether a safety car is likely to be deployed on a given
lap, using weather conditions, tire data, and telemetry-based anomaly
detection (sudden speed drops, sudden pace drops).

BIG PICTURE STEPS:
1. Pull race data from FastF1 (laps, weather, race control messages)
2. Build features for each lap (weather, tire age, pace trends, etc.)
3. Build the target variable: was a safety car active on this lap?
4. Engineer extra features: telemetry anomalies + weather deviations
5. Split into training data and testing data BY RACE (not randomly!)
   so we can test if the model generalizes to tracks it's never seen
6. Train a Random Forest classifier
7. Evaluate: how well did it actually predict safety cars?
8. Check feature importance: which factors mattered most?
==========================================================================
"""

import fastf1
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use("Agg")  # non-interactive backend - just saves plots to files, no popup windows
import matplotlib.pyplot as plt


# ==========================================================================
# STEP 0: SET UP CACHING
# ==========================================================================
# FastF1 downloads real F1 data from the internet the first time you ask
# for a session. That's slow and could hit rate limits if done every run.
# Caching saves the downloaded data locally so future runs are instant.

base_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(base_dir, 'fastf1_cache')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)


# ==========================================================================
# STEP 1: TRACK ABRASION LOOKUP TABLE
# ==========================================================================
# "Abrasion" = how rough/harsh a track surface is on tires. This isn't
# something FastF1 gives us directly, so we hardcode it ourselves based
# on real-world knowledge (Pirelli's own difficulty ratings, tire wear
# reports, etc). Higher number = more abrasive = wears tires out faster.

TRACK_ABRASION = {
    # High Abrasion (4-5) - Destroys tires quickly
    "Bahrain Grand Prix": 5,
    "Japanese Grand Prix": 5,
    "Spanish Grand Prix": 4,
    "British Grand Prix": 4,
    "Belgian Grand Prix": 4,
    "United States Grand Prix": 4,
    "Brazilian Grand Prix": 4,
    "Dutch Grand Prix": 4,
    "Qatar Grand Prix": 4,

    # Medium Abrasion (3) - Balanced wear
    "French Grand Prix": 3,
    "Austrian Grand Prix": 3,
    "Hungarian Grand Prix": 3,
    "Italian Grand Prix": 3,  # Monza
    "Abu Dhabi Grand Prix": 3,
    "Emilia Romagna Grand Prix": 3,  # Imola
    "Portuguese Grand Prix": 3,

    # Low Abrasion (1-2) - Very smooth or slippery
    "German Grand Prix": 2,
    "Canadian Grand Prix": 2,
    "Russian Grand Prix": 2,
    "Mexico City Grand Prix": 2,
    "Saudi Arabian Grand Prix": 2,
    "Monaco Grand Prix": 1,
    "Azerbaijan Grand Prix": 1,
    "Turkish Grand Prix": 1,
    "Singapore Grand Prix": 1
}


def track_abrasion_score(session_name):
    """Look up how abrasive a track is, based on its name.
    If we don't have it in our table, just guess "medium" (3)."""
    for key in TRACK_ABRASION:
        if key in session_name:
            return TRACK_ABRASION[key]
    return 3  # default guess if track isn't in our list


# ==========================================================================
# STEP 2: FIND WHICH LAPS HAD A SAFETY CAR (this becomes our target variable)
# ==========================================================================

def get_safety_car_laps(session):
    """
    FastF1 gives us "race control messages" - official announcements sent
    out during the race (like a live news ticker). Some of these say things
    like "SAFETY CAR DEPLOYED". This function finds those messages and
    figures out which lap numbers they happened around.

    Problem: messages have a real-world timestamp (like "3:15 PM"), but
    laps are stored as "time since the session started" (like "1 hour
    20 minutes in"). So we have to convert between the two before we can
    compare them.
    """
    sc_laps = set()  # will hold every lap number where a safety car was active

    messages = session.race_control_messages

    if messages is None or len(messages) == 0:
        return sc_laps  # no messages at all, nothing to check

    # this is the real-world clock time the session started - our conversion anchor
    session_start = session.t0_date

    # go through every single race control message one at a time
    for i in range(len(messages)):
        msg_text = str(messages.iloc[i]["Message"]).upper()

        if "SAFETY CAR" not in msg_text:
            continue  # not a safety car message, skip it and check the next one

        # convert this message's real-world time into "time since session start",
        # so it's the same format as the lap times we'll compare it to
        msg_time = messages.iloc[i]["Time"]
        msg_time_as_timedelta = msg_time - session_start

        # now check EVERY lap, from EVERY driver, to see if it happened
        # close in time to this safety car message
        for j in range(len(session.laps)):
            lap_time = session.laps.iloc[j]["Time"]
            lap_num = session.laps.iloc[j]["LapNumber"]

            time_diff = (lap_time - msg_time_as_timedelta).total_seconds()

            # "close enough" = up to 2 minutes before the message,
            # or up to 30 seconds after it
            if -120 <= time_diff <= 30:
                sc_laps.add(lap_num)

    return sc_laps


# ==========================================================================
# STEP 3: BUILD THE FULL FEATURE TABLE FOR ONE RACE
# ==========================================================================

def build_feature_table(year, event_name):
    """
    This is the main function that builds one race's worth of data.
    It returns a table with one row per driver per lap, with all our
    features (weather, tire age, pace trends) and our target variable
    (SafetyCar: 1 or 0).
    """

    # --- load the actual race session from FastF1 ---
    session = fastf1.get_session(year, event_name, "R")  # "R" = the main Race
    session.load(telemetry=True, weather=True)

    # drop laps with no lap time at all (usually in/out laps, pit laps)
    laps = session.laps.dropna(subset=["LapTime"]).copy()

    # convert lap time from a "duration" format into plain seconds,
    # much easier to do math with
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

    # --- attach weather data to each lap ---
    # weather is recorded separately from laps, so for each lap we find
    # whichever weather reading was closest in time to that lap
    weather = session.weather_data

    track_temps = []
    air_temps = []
    rainfalls = []
    humiditys = []
    wind_speeds = []

    for i in range(len(laps)):
        lap_time = laps.iloc[i]["Time"]

        # find the weather reading with the smallest time difference to this lap
        time_diffs = abs(weather["Time"] - lap_time)
        closest_index = time_diffs.idxmin()
        closest_weather = weather.loc[closest_index]

        track_temps.append(closest_weather["TrackTemp"])
        air_temps.append(closest_weather["AirTemp"])
        rainfalls.append(closest_weather["Rainfall"])
        humiditys.append(closest_weather["Humidity"])
        wind_speeds.append(closest_weather["WindSpeed"])

    laps["TrackTemp"] = track_temps
    laps["AirTemp"] = air_temps
    laps["Rainfall"] = rainfalls
    laps["Humidity"] = humiditys
    laps["WindSpeed"] = wind_speeds

    # every lap in this race gets the same track abrasion score
    laps["TrackAbrasion"] = track_abrasion_score(event_name)

    # --- calculate each driver's rolling average lap time ---
    # this measures "how has this driver been performing recently"
    # using their last 3 laps as a rolling window
    laps = laps.sort_values(["Driver", "LapNumber"])

    rolling_avgs = []
    drivers = laps["Driver"].unique()

    for driver in drivers:
        driver_laps = laps[laps["Driver"] == driver]
        lap_times = driver_laps["LapTimeSeconds"].tolist()

        recent = []  # keeps track of the last few lap times as we go
        for time in lap_times:
            recent.append(time)
            if len(recent) > 3:
                recent.pop(0)  # only keep the most recent 3 laps
            avg = sum(recent) / len(recent)
            rolling_avgs.append(avg)

    laps["RollingAvg"] = rolling_avgs
    # how much faster/slower was this lap compared to the driver's recent average?
    laps["LapTimeDelta"] = laps["LapTimeSeconds"] - laps["RollingAvg"]

    # --- build our target variable: was a safety car active this lap? ---
    sc_laps = get_safety_car_laps(session)
    safety_car_column = []
    for i in range(len(laps)):
        lap_num = laps.iloc[i]["LapNumber"]
        if lap_num in sc_laps:
            safety_car_column.append(1)
        else:
            safety_car_column.append(0)
    laps["SafetyCar"] = safety_car_column

    # tag every row with which race it came from - important later when
    # we split data by race instead of randomly
    laps["Race"] = str(year) + " " + event_name

    return laps


# ==========================================================================
# STEP 4: PULL DATA FOR A WHOLE LIST OF RACES
# ==========================================================================

# these are raw FastF1 columns we don't actually use anywhere in our model,
# just clutter we can drop to keep the table cleaner
columns_to_drop = [
    "DriverNumber", "PitOutTime", "PitInTime",
    "Sector1Time", "Sector2Time", "Sector3Time",
    "Sector1SessionTime", "Sector2SessionTime", "Sector3SessionTime",
    "IsPersonalBest", "FreshTyre", "LapStartTime", "LapStartDate",
    "Deleted", "DeletedReason", "FastF1Generated", "IsAccurate",
    "TrackStatus",
]

# instead of hand-typing every race, pull every race from these two
# full seasons automatically - gives us more variety in tracks/weather
SEASONS_TO_PULL = [2019, 2021]

RACES = []

for season in SEASONS_TO_PULL:
    schedule = fastf1.get_event_schedule(season)
    for _, event in schedule.iterrows():
        round_number = event["RoundNumber"]
        event_name = event["EventName"]
        if round_number == 0:
            continue  # round 0 is pre-season testing, not a real race, skip it
        RACES.append((season, event_name))

print("Total races to pull:", len(RACES))

all_laps = []  # will collect one dataframe per race, combined at the end

for year, event_name in RACES:
    print("Trying", year, event_name)
    try:
        df = build_feature_table(year, event_name)
        all_laps.append(df)
        num_sc = df["SafetyCar"].sum()
        print("Got", len(df), "laps,", num_sc, "were under safety car")
    except Exception as e:
        # if one race fails to load (missing data, weird format, etc),
        # don't let it crash the whole pull - just skip it and move on
        print("Skipping", year, event_name, "- error:", e)

# combine every race's data into one big table
data = pd.concat(all_laps)
data = data.drop(columns=columns_to_drop)
data = data.reset_index(drop=True)

print("\nTotal laps:", len(data))
print("Total safety car laps:", data["SafetyCar"].sum())

# save this raw combined table so we don't have to re-download everything
# if something later in the script goes wrong
data.to_csv("safety_car_combined.csv", index=False)
print("Saved to safety_car_combined.csv")


# ==========================================================================
# STEP 5: TELEMETRY-BASED ANOMALY FEATURES
# ==========================================================================
# Idea: weather alone tells us about background CONDITIONS, but doesn't
# capture the actual EVENT (a crash, a spin) that triggers a safety car.
# Speed trap data lets us detect "this driver suddenly went much slower
# than their own normal pace" - a strong sign something just happened.

# some speed trap readings can be missing (e.g. car crashed before reaching
# the trap) - fill missing values with 0 so we can do math on them safely
for col in ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

# the fastest speed this driver hit anywhere on track this lap
data["Max_Lap_Speed"] = data[["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]].max(axis=1)

# each driver's "normal" top speed for this race so far (a running average,
# ignoring laps where speed was 0 e.g. a formation lap or red flag)
data["Driver_Avg_Top_Speed"] = data.groupby(["Race", "Driver"])["Max_Lap_Speed"].transform(
    lambda s: s.replace(0, pd.NA).expanding().mean()
)

# how much SLOWER was this lap's top speed compared to their own normal?
# a big positive number here = something unusual happened (crash, spin,
# mechanical issue) - much stronger and more direct than weather alone
data["Speed_Anomaly_Delta"] = data["Driver_Avg_Top_Speed"] - data["Max_Lap_Speed"]
data["Speed_Anomaly_Delta"] = data["Speed_Anomaly_Delta"].fillna(0)  # fill early-lap gaps

# "previous lap" pace anomaly - using the PREVIOUS lap (not the current one)
# avoids a subtle cheat: cars are forced to slow down once a safety car is
# already out, so using the current lap's slowdown to predict the safety
# car would partly just be detecting the safety car's own effect
data = data.sort_values(["Race", "Driver", "LapNumber"])
data["Previous_LapTimeDelta"] = data.groupby(["Race", "Driver"])["LapTimeDelta"].shift(1)

# --- track-wide danger signal ---
# a safety car affects the WHOLE FIELD, not just one driver. So instead of
# only looking at one driver's anomaly, we check: out of ALL drivers on
# track this lap, what's the BIGGEST anomaly anyone showed? If one driver
# crashes, this should spike even for drivers who didn't personally crash.
data["Trackwide_Max_Speed_Anomaly"] = data.groupby(["Race", "LapNumber"])["Speed_Anomaly_Delta"].transform("max")
data["Trackwide_Max_Pace_Drop"] = data.groupby(["Race", "LapNumber"])["Previous_LapTimeDelta"].transform("max")


# ==========================================================================
# STEP 6: WEATHER DEVIATION FEATURES (relative, not absolute)
# ==========================================================================
# Problem with raw weather values: "60% humidity" might be normal at one
# track and highly unusual at another. Instead, we measure how far each
# lap's weather is from THAT RACE's own average - "unusual for this race"
# rather than one fixed number expected to mean the same thing everywhere.

data["Humidity_RaceAvg"] = data.groupby("Race")["Humidity"].transform("mean")
data["Humidity_Deviation"] = data["Humidity"] - data["Humidity_RaceAvg"]

data["TrackTemp_RaceAvg"] = data.groupby("Race")["TrackTemp"].transform("mean")
data["TrackTemp_Deviation"] = data["TrackTemp"] - data["TrackTemp_RaceAvg"]

data["AirTemp_RaceAvg"] = data.groupby("Race")["AirTemp"].transform("mean")
data["AirTemp_Deviation"] = data["AirTemp"] - data["AirTemp_RaceAvg"]

data["WindSpeed_RaceAvg"] = data.groupby("Race")["WindSpeed"].transform("mean")
data["WindSpeed_Deviation"] = data["WindSpeed"] - data["WindSpeed_RaceAvg"]


# ==========================================================================
# STEP 7: BUILD X (inputs) AND y (target) FOR THE MODEL
# ==========================================================================

features = [
    "TyreLife", "TrackTemp_Deviation", "AirTemp_Deviation",
    "Rainfall", "Humidity_Deviation", "WindSpeed_Deviation",
    "TrackAbrasion",
    "Previous_LapTimeDelta", "Trackwide_Max_Pace_Drop",
    "Speed_Anomaly_Delta", "Trackwide_Max_Speed_Anomaly"
]

# drop any row missing a value in any of our chosen features or the target -
# the model can't handle missing values
model_data = data.dropna(subset=features + ["SafetyCar"]).copy()

# Compound (SOFT/MEDIUM/HARD/etc) is text, but models need numbers.
# one-hot encoding turns it into several 0/1 columns, one per tire type
compound_dummies = pd.get_dummies(model_data["Compound"], prefix="Compound")

X = pd.concat([model_data[features], compound_dummies], axis=1)  # all our inputs
y = model_data["SafetyCar"]  # the answer we're trying to predict

# quick sanity check: how does each individual feature correlate with
# SafetyCar on its own? (simple, straight-line relationships only -
# doesn't capture more complex patterns the model itself might find)
print("\n--- Correlation with SafetyCar (using deviation features) ---")
correlation_check = X.copy()
correlation_check["SafetyCar"] = y
print(correlation_check.corr()["SafetyCar"].sort_values(ascending=False))


# ==========================================================================
# STEP 8: TRAIN / TEST SPLIT - BY RACE, NOT RANDOMLY
# ==========================================================================
# Why not just randomly split rows? Because laps from the SAME race share
# a lot of context (same weather day, same track, same specific safety car
# event). If some laps from a race end up in training and others from that
# SAME race end up in testing, the model could partly "cheat" by having
# already seen that exact situation. Splitting by whole races instead
# means the model has NEVER seen anything from the test races at all -
# a much more honest test of whether it generalizes to new situations.

test_races = ["2019 German Grand Prix", "2020 Turkish Grand Prix"]

print(f"\nTraining on: {[r for r in model_data['Race'].unique() if r not in test_races]}")
print(f"Testing on: {test_races}")

train_data = model_data[~model_data["Race"].isin(test_races)]
test_data = model_data[model_data["Race"].isin(test_races)]

print("Training rows:", len(train_data))
print("Testing rows:", len(test_data))

# rebuild X/y separately for train and test (compound dummies need to be
# built per-set, since not every tire compound may appear in both)
X_train = train_data[features]
compound_dummies_train = pd.get_dummies(train_data["Compound"], prefix="Compound")
X_train = pd.concat([X_train, compound_dummies_train], axis=1)
y_train = train_data["SafetyCar"]

X_test = test_data[features]
compound_dummies_test = pd.get_dummies(test_data["Compound"], prefix="Compound")
X_test = pd.concat([X_test, compound_dummies_test], axis=1)
y_test = test_data["SafetyCar"]

# make sure train and test have EXACTLY the same columns in the same order -
# if one compound type appears in train but not test (or vice versa),
# this fills in a 0 column so the model doesn't break
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)


# ==========================================================================
# STEP 9: TRAIN THE MODEL
# ==========================================================================

clf = RandomForestClassifier(
    n_estimators=200,        # how many individual decision trees to build
    max_depth=7,              # how many questions deep each tree can go
                               # (kept shallow to reduce overfitting/memorizing)
    random_state=42,          # makes results reproducible - same result every run
    class_weight="balanced"   # safety car laps are rare - this tells the model
                               # to penalize getting them wrong more heavily,
                               # instead of just always guessing "no safety car"
)
clf.fit(X_train, y_train)


# ==========================================================================
# STEP 10: EVALUATE THE MODEL
# ==========================================================================

# get a probability (0 to 1) for each test lap, not just a hard yes/no
y_prob = clf.predict_proba(X_test)[:, 1]

# instead of a fixed threshold, use whatever risk score sits at the 80th
# percentile of THIS test set - i.e. "flag the riskiest 20% of laps"
custom_threshold = np.percentile(y_prob, 80)
y_pred = (y_prob > custom_threshold).astype(int)

print(f"\n--- Model evaluation at dynamic threshold {custom_threshold * 100:.1f}% ---")
print(classification_report(y_test, y_pred, zero_division=0))

# ROC-AUC measures whether the model's risk scores are correctly RANKED -
# does it generally score real safety car laps higher than normal laps,
# regardless of where exactly we draw the yes/no cutoff line?
# 0.5 = no better than random guessing. 1.0 = perfect ranking.
auc_score = roc_auc_score(y_test, y_prob)
print(f"\nROC-AUC score: {auc_score:.3f} (Aiming for > 0.65)")


# ==========================================================================
# STEP 11: FEATURE IMPORTANCE - WHICH FACTORS MATTERED MOST?
# ==========================================================================

importances = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print("\n--- Feature importance (with deviation features) ---")
print(importances)

plt.figure(figsize=(8, 5))
importances.plot(kind="barh")
plt.gca().invert_yaxis()  # highest importance at the top of the chart
plt.title("What predicts a safety car? (deviation features)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance_deviation.png", dpi=150)
print("\nSaved plot to feature_importance_deviation.png")