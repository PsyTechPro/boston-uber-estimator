from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = FastAPI()

# =========================
# ENABLE CORS (FIXES YOUR ISSUE)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD + TRAIN (runs once at startup)
# =========================
df = pd.read_csv("boston_rides.csv")

df = df[['timestamp', 'hour', 'distance', 'price', 'name']]
df = df.dropna()
df = df[df['name'] == 'UberX']
df = df[df['distance'] > 0]

df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
df['day_of_week'] = df['datetime'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

df['price_per_mile'] = df['price'] / df['distance']

# =========================
# PRICE MODEL
# =========================
X_price = df[['hour', 'day_of_week', 'distance', 'is_weekend']]
y_price = df['price_per_mile']

price_model = RandomForestRegressor(n_estimators=100)
price_model.fit(X_price, y_price)

# =========================
# DEMAND LOOKUP TABLE
# =========================
demand = df.groupby(['hour', 'day_of_week']).size().reset_index(name='rides')
demand['demand_score'] = demand['rides'] / demand['rides'].max()

# =========================
# API ROUTE
# =========================
@app.get("/predict")
def predict(
    day: int = Query(..., description="Day of week (0=Mon, 1=Tue, ..., 6=Sun)"),
    hour: int = Query(..., description="Hour of day (0=midnight, 13=1pm, 23=11pm)"),
    distance: float = Query(..., description="Trip distance in miles")
):

    is_weekend = 1 if day >= 5 else 0

    # =========================
    # PRICE
    # =========================
    price_per_mile = price_model.predict([[hour, day, distance, is_weekend]])[0]
    base_price = price_per_mile * distance

    # =========================
    # DEMAND LOOKUP
    # =========================
    match = demand[
        (demand['hour'] == hour) &
        (demand['day_of_week'] == day)
    ]

    if len(match) > 0:
        demand_score = match['demand_score'].values[0]
    else:
        demand_score = 0.3

    # =========================
    # DEMAND CLASSIFICATION
    # =========================
    if demand_score < 0.2:
        demand_level = "Low"
        wait_time = "1-2 minutes"
    elif demand_score < 0.5:
        demand_level = "Moderate"
        wait_time = "3-5 minutes"
    else:
        demand_level = "High"
        wait_time = "6-10+ minutes"

    # =========================
    # SURGE (REALISTIC RULES)
    # =========================
    surge = 1.0

    # Friday & Saturday evening (6pm+)
    if day in [4, 5] and hour >= 18:
        surge = 1.5

    # Late night peak (11pm+ Fri/Sat)
    if day in [4, 5] and hour >= 23:
        surge = 2.0

    # Weekday rush hours
    elif day in [0, 1, 2, 3] and (7 <= hour <= 9 or 16 <= hour <= 18):
        surge = 1.2

    # Sunday evening
    elif day == 6 and 17 <= hour <= 21:
        surge = 1.3

    final_price = base_price * surge

    # =========================
    # TRIP TIME
    # =========================
    if 7 <= hour <= 9:
        speed = 15
    elif 16 <= hour <= 18:
        speed = 15
    elif hour >= 22 or hour <= 5:
        speed = 30
    else:
        speed = 22

    if is_weekend == 1:
        speed += 3

    trip_time = (distance / speed) * 60

    # =========================
    # OUTPUT
    # =========================
    return {
        "base_price": round(base_price, 2),
        "surge_multiplier": surge,
        "final_price": round(final_price, 2),
        "demand": demand_level,
        "wait_time": wait_time,
        "trip_time_minutes": int(trip_time)
    }