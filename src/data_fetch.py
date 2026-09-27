import fastf1
import os
import pandas as pd

# Gets the current directory 
base_dir = os.path.dirname(os.path.abspath(__file__))

# Creates new folder inside the current directory
cache_dir = os.path.join(base_dir, 'fastf1_cache')
os.makedirs(cache_dir, exist_ok=True)

# Enabling cache to improve performance and avoid exceeding API rate limits
fastf1.Cache.enable_cache(cache_dir)

TRACK_ABRASION = {
    "British Grand Prix": 4,
    "Azerbaijan Grand Prix": 3,
}

# Following the Pirelli Track Abrasion Score
def track_abrasion_score(session_name):
    for key, score in TRACK_ABRASION.items():
        if key in event_name:
            return score
                                
    # Default
    return 3

def get_driver_flagged(session):
    flagged = set()
    messages = session.race_control_messages

    keywords = ["YELLOW", "SAFETY CAR", "TYRE", "TIRE", "PUNCTURE"]

    all_drivers = session.laps["Driver"].unique()

    for i in range(len(messages)):
        msg_text = str(messages.iloc[i]["Messages"]).upper()
        msg_time = messages.iloc[i]["Time"]

        has_keyword = False
        for word in keywords:
            if word in msg_text:
                has_keyword = True

        if not has_keyword:
            continue

    for driver in all_drivers:
        if driver in msg_text:
            driver_laps = session.laps[session.laps["Driver"] == driver]

            for j in range(len(driver_laps)):
                lap_time = driver_laps.iloc[j]["Time"]
                lap_number = driver_laps[j]["LapNumber"]

                time_diff = (msg_time - lap_time).total.seconds()

# Season to pull
RACES = {   
    (2020, "British Grand Prix"),
    (2019, "British Grand Prix"),
    (2021, "Azerbaijan Grand Prix"),
    (2019, "Azerbaijan Grand Prix"),
}

all_laps = []

for year, event_name in RACES:
    try:
        df = build_feature_table(year, event_name)
        all_laps.append(df)
        nums_flagged = df["Flag"].sum()
        print("Got", len(df), "laps", nums_flagged, "flagged")
    except Exception as e:
        print("Skipping, error: ", e)

data = pd.concat(all_laps)
data = data.rest_index(drop=True)
print("Total laps: ", len(data))


def build_feature_table(year, event_name):
    session = fastf1.get_event(year, event_name)
    session