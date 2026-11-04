#Line 1-2: Imports pandas (used for loading and manipulating table data) and LogisticRegression
#from scikit-learn (the machine learning algorithm used to calculate shot probabilities).
import pandas as pd
from sklearn.linear_model import LogisticRegression

#Line 8-9: Adjusts pandas settings so when dataframes print to the Thonny terminal, columns don't
#wrap around or get truncated with ...
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

#Line 13-15: Stores the exact file paths to variables using raw string notation (r"..."), which prevents
#Windows backslashes (\) from causing escape character errors.
train_path = r"C:\training.csv"
test_path = r"C:testing.csv\testing.csv"
sub_path = r"C:\submission.csv"

#Line 19-21: Prints a status message and uses pd.read_csv() to decompress and read the training and
#testing datasets directly into pandas DataFrames in memory.
print("Loading data...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

#Line 25: Defines a list of the numerical column names chosen as input features for the model
#(shot distance from basket, distance of closest defender, and seconds remaining on shot clock).
features = ['distance', 'closestdefdist', 'shotclock']

#Line 31–32:
#X_train: Extracts only the specified feature columns from the training data and fills
#any missing values (NaN) with 0 so the model doesn't error out.
#Y_train: Isolates the ground truth target column (1 for made shot, 0 for missed shot).
X_train = train_df[features].fillna(0)
Y_train = train_df['outcome']

#Line 36–38: Initializes a Logistic Regression model object and trains it on the input features
#(X_train) against the actual shot outcomes (Y_train).
print("Training model...")
model = LogisticRegression()
model.fit(X_train, Y_train)

#Line 44–46:
#Extracts and cleans the same feature columns from the testing set.
#model.predict_proba(X_test)[:, 1] calculates probability values between 0.0 and 1.0
#(where [:, 1] extracts class 1, the probability the shot is made) and assigns them to the 'outcome' column in test_df.
print("Predicting test probabilities...\n")
X_test = test_df[features].fillna(0)
test_df['outcome'] = model.predict_proba(X_test)[:, 1]

#Line 51-52:
#Selects just shot_id and outcome, sorting rows from highest probability to lowest (ascending=False).
#to_csv(..., index=False) exports the result to submission.csv without writing pandas' automatic row numbers (0, 1, 2...).
submission = test_df[['shot_id', 'outcome']].sort_values(by='outcome', ascending=False)
submission.to_csv(sub_path, index=False)

#Line 55: Creates a separate copy of the testing dataframe with key features included to build a formatted display in the terminal.
preview_df = test_df[['shot_id', 'distance', 'closestdefdist', 'shotclock', 'outcome']].copy()

#Line 59-62: Formats the numerical values in the display table: rounds distances and shot clock to 1 decimal place with units
#(ft, s) and converts probabilities into readable percentages (99.8%).
preview_df['distance']       = preview_df['distance'].round(1).astype(str) + ' ft'
preview_df['closestdefdist'] = preview_df['closestdefdist'].round(1).astype(str) + ' ft'
preview_df['shotclock']      = preview_df['shotclock'].round(1).astype(str) + ' s'
preview_df['outcome']        = (preview_df['outcome'] * 100).round(1).astype(str) + '%'

#Line 65: Renames column headers to clean, human-readable labels for the terminal display.
preview_df.columns = ['Shot ID', 'Shot Dist', 'Def Dist', 'Shot Clock', 'Make Prob']

#Line 69–78: Prints visual section headers and outputs the 5 highest and 5 lowest shot probabilities alongside their shot context.
#.to_string(index=False) hides pandas row indices.
divider = "=" * 55
print(divider)
print("TOP 5 MOST LIKELY SHOTS MADE")
print(divider)
print(preview_df.sort_values(by='Make Prob', ascending=False).head().to_string(index=False))

print("\n" + divider)
print("TOP 5 LEAST LIKELY SHOTS MADE")
print(divider)
print(preview_df.sort_values(by='Make Prob', ascending=True).head().to_string(index=False))

#Line 82–89: Prints a preview of the first 10 rows formatted as they appear inside submission.csv (rounded to 4 decimal places)
#and outputs a final confirmation message.
print("\n" + divider)
print("FIRST 10 SUBMISSION PREDICTIONS")
print(divider)
formatted_sub = submission.head(10).copy()
formatted_sub['outcome'] = formatted_sub['outcome'].round(4)
print(formatted_sub.to_string(index=False))

print("\nProcess complete! Saved results to 'submission.csv'.")
