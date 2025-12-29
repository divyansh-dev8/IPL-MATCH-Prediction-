import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =====================
# LOAD DATA
# =====================
data = pd.read_csv("data/matches.csv")

# Fix season (2007/08 → 2007)
data['season'] = data['season'].astype(str).str[:4].astype(int)

# Required columns only
data = data[['team1', 'team2', 'toss_decision', 'venue', 'season', 'winner']]

# 🎯 Create target (IMPORTANT)
data['winner_team1'] = (data['winner'] == data['team1']).astype(int)

# Features & target
X = data[['team1', 'team2', 'toss_decision', 'venue', 'season']]
y = data['winner_team1']
# =====================
# PREPROCESSING
# =====================
categorical_cols = ['team1', 'team2', 'toss_decision', 'venue']
numeric_cols = ['season']

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            ['team1', 'team2', 'toss_decision', 'venue']
        ),
        (
            'num',
            'passthrough',
            ['season']
        )
    ]
)
# =====================
# MODEL PIPELINE
# =====================
model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# =====================
# TRAIN TEST SPLIT
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# =====================
# ACCURACY
# =====================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("✅ Model Accuracy:", accuracy)

# =====================
# PREDICTION FUNCTION
# =====================
def predict_match(team1, team2, toss, venue, season):
    input_df = pd.DataFrame([{
        'team1': team1,
        'team2': team2,
        'toss_decision': toss,
        'venue': venue,
        'season': season
    }])
    return model.predict(input_df)

# =====================
# USER INPUT
# =====================
print("\n===== USER INPUT IPL MATCH PREDICTION =====")

team1 = input("Enter Team 1: ")
team2 = input("Enter Team 2: ")
toss = input("Toss decision (bat/field): ")
venue = input("Enter Venue: ")
season = int(input("Enter Season (year): "))

winner_pred = predict_match(team1, team2, toss, venue, season)

if winner_pred[0] == 1:
    print(f"\n🏏 PREDICTED WINNER: {team1}")
else:
    print(f"\n🏏 PREDICTED WINNER: {team2}")