# Code you have previously used to load data
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = 'MLExample.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.Salary
feature_columns = ['Age']
X = home_data[feature_columns]

# Specify Model
iowa_model = DecisionTreeRegressor()
# Fit Model
iowa_model.fit(X, y)

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Test")
print("Actual target values for those homes:", y.head().tolist())