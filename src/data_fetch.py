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
    "Bahrain Grand Prix": 3,
    "Azerbaijan Grand Prix": 3,
}

# Following the Pirelli Track Abrasion Score
def track_abrasion_score(session_name):
    for key, score in TRACK_ABRASION.items():
        if key in session_name:
            return score
                                
    # Default
    return 3

def get_safety_car_laps(session):
    sc_laps = set()

    messages = session.race_control_messages

    if messages is None or len(messages) == 0:
        return sc_laps

    session_start = session.t0_date

    for i in range(len(messages)):
        msg_text = str(messages.iloc[i]["Message"]).upper()

        if "SAFETY CAR" not in msg_text:
            continue

        msg_time = messages.iloc[i]["Time"]
        msg_time_as_timedelta = msg_time - session_start

        for j in range(len(session.laps)):
            lap_time = session.laps.iloc[j]["Time"]
            lap_num = session.laps.iloc[j]["LapNumber"]

            time_diff = (lap_time - msg_time_as_timedelta).total_seconds()

            if -120 <= time_diff <= 30:
                sc_laps.add(lap_num)

    return sc_laps
 
def build_feature_table(year, event_name):
    session = fastf1.get_session(year, event_name, "R")
    session.load(telemetry=True, weather=True)
 
    laps = session.laps.dropna(subset=["LapTime"]).copy()
 
    # convert lap time to plain seconds so its easier to work with
    laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()
 
    weather = session.weather_data
 
    track_temps = []
    air_temps = []
    rainfalls = []
    humiditys = []
    wind_speeds = []
 
    for i in range(len(laps)):
        lap_time = laps.iloc[i]["Time"]
 
        # find the closest weather reading to this lap's time
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
 
    # race progress and fuel estimate
    total_laps = laps["LapNumber"].max()
    race_progress = []
    fuel_estimate = []
    for i in range(len(laps)):
        lap_num = laps.iloc[i]["LapNumber"]
        progress = lap_num / total_laps
        race_progress.append(progress)
        fuel_estimate.append(1 - progress)
 
    laps["RaceProgressPct"] = race_progress
    laps["FuelLoadEstimate"] = fuel_estimate
 
    # track abrasion, same number for every lap in this race
    laps["TrackAbrasion"] = track_abrasion_score(event_name)
 
    # lap time delta compared to that driver's last 3 laps
    laps = laps.sort_values(["Driver", "LapNumber"])
    rolling_avgs = laps.groupby("Driver")["LapTimeSeconds"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    laps["RollingAvg"] = rolling_avgs
    laps["LapTimeDelta"] = laps["LapTimeSeconds"] - laps["RollingAvg"]
 
    # target variable: was a safety car out on this lap
    sc_laps = get_safety_car_laps(session)
    safety_car_column = []
    for i in range(len(laps)):
        lap_num = laps.iloc[i]["LapNumber"]
        if lap_num in sc_laps:
            safety_car_column.append(1)
        else:
            safety_car_column.append(0)
    laps["SafetyCar"] = safety_car_column
 
    laps["Race"] = str(year) + " " + event_name
    return laps
 
 
# list of races we want to use
RACES = [
    (2020, "British Grand Prix"),
    (2019, "British Grand Prix"),
    (2022, "British Grand Prix"),
    (2021, "Azerbaijan Grand Prix"),
    (2019, "Azerbaijan Grand Prix"),
    (2022, "Azerbaijan Grand Prix"),
]
 
all_laps = []
 
for year, event_name in RACES:
    df = build_feature_table(year, event_name)
    all_laps.append(df)
    num_sc = df["SafetyCar"].sum()
    print("Got", len(df), "laps,", num_sc, "were under safety car") 

data = pd.concat(all_laps)
data = data.reset_index(drop=True)
 
print("\nTotal laps:", len(data))
print("Total safety car laps:", data["SafetyCar"].sum())
 
data.to_csv("safety_car_combined.csv", index=False)
print("Saved to safety_car_combined.csv")