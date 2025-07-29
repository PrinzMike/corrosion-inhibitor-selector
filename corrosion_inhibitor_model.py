# corrosion_inhibitor_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("/Users/MICHAEL/Desktop/corrosion-inhibitor-selector/corrosion_inhibitor_dataset.csv")

# ===============================
# 2. Extract Averages from Ranges
# ===============================
def extract_avg(value):
    try:
        if pd.isna(value):
            return np.nan
        value = str(value)
        if "→" in value:
            parts = value.split("→")
            return np.mean([float(p.strip()) for p in parts])
        elif "–" in value or "-" in value:
            parts = value.replace("–", "-").split("-")
            return np.mean([float(p.strip()) for p in parts])
        elif "up to" in value.lower():
            return float(value.lower().replace("up to", "").strip())
        elif value.lower() in ["n/a", "na"]:
            return np.nan
        else:
            return float(value.strip())
    except:
        return np.nan

# Apply cleaning
df["Concentration (ppm)"] = df["Concentration (mg/L or ppm)"].apply(extract_avg)
df["Temperature (°C)"] = df["Temperature (°C)"].apply(extract_avg)
df["Exposure Time (h)"] = df["Exposure Time (h)"].apply(extract_avg)
df["Inhibition Efficiency (%)"] = df["Inhibition Efficiency (%)"].apply(extract_avg)


# 3. Drop Unused Columns

df_clean = df.drop(columns=["Concentration (mg/L or ppm)", "Notes"])


# 4. Encode Categorical Columns

le_inhibitor = LabelEncoder()
le_source = LabelEncoder()

df_clean["Inhibitor_Code"] = le_inhibitor.fit_transform(df_clean["Inhibitor"].astype(str))
df_clean["Source_Code"] = le_source.fit_transform(df_clean["Source"].astype(str))


# 5. Prepare Model Data

df_model = df_clean[[
    "Inhibitor_Code", "Source_Code", "Concentration (ppm)",
    "Temperature (°C)", "Exposure Time (h)", "Inhibition Efficiency (%)"
]]


# 6. Fill Missing Values Safely

df_model = df_model.fillna(-1)


# 7. Train/Test Split

X = df_model.drop(columns=["Inhibition Efficiency (%)"])
y = df_model["Inhibition Efficiency (%)"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 8. Train Random Forest Model

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# 9. Evaluate Model

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("✅ Model trained successfully!")
print(f"📉 RMSE: {rmse:.2f}")
print(f"📈 R² Score: {r2:.2f}")


# 10. Save Model and Encoders

joblib.dump(model, "corrosion_model.pkl")
joblib.dump(le_inhibitor, "le_inhibitor.pkl")
joblib.dump(le_source, "le_source.pkl")
