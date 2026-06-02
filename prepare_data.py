# prepare_data.py
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load raw data
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["MedHouseVal"] = housing.target

# --- Carry out the Feature Engineering ---
# 1. Rooms per household
df["RoomsPerHousehold"] = df["AveRooms"] / df["AveOccup"]

# 2. Bedrooms per room
df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"]

# 3. Population per household
df["PopulationPerHousehold"] = df["Population"] / df["AveOccup"]

# 4. Log transform of skewed features (example: MedInc often benefits)
df["LogMedInc"] = np.log1p(df["MedInc"])

# Drop rows with infinity or NaN (caused by division by zero)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Split into train / test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save processed datasets
train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)

print(f"Training set: {train_df.shape}")
print(f"Test set: {test_df.shape}")
print("Columns after engineering:", list(df.columns))
