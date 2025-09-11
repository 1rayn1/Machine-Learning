import pandas as pd #used for reading and data manipulation
from sklearn.metrics import mean_absolute_error #used to measure model performance
from sklearn.model_selection import train_test_split #used to split data into training and validation sets
from sklearn.tree import DecisionTreeRegressor #used to create the model


# Path of the file to read
example_file_path = 'car data.csv'

try:
    #This reads the data into a pandas DataFrame
    example_data = pd.read_csv(example_file_path, index_col=0) #index_col=0 uses the first column as the index
    # Display the shape of the data in (rows, columns)
    example_data.shape
    # Display the first five lines of the data
    example_data.head()

    if example_data.isnull().any().any():
        raise ValueError("Data contains missing values. Please clean or impute the data before modeling.")

    # Create target object and call it y
    y = example_data.Present_Price
    # Create X(The feature(s) used to make the prediction)
    features = ['Year','Selling_Price','Kms_Driven']
    X = example_data[features]

    # Split into validation and training data(train_X and train_y are the training data, while val_X and val_y are the validation data)
    #random_state is set to 1 to ensure reproducibility
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    # Creates the model.
    # Setting random_state to 1 to ensure same results each run
    iowa_model = DecisionTreeRegressor(random_state=1)
    # Fits the model into the training data
    iowa_model.fit(train_X, train_y)
    #predict() function predicts the salaries for the validation data
    val_predictions = iowa_model.predict(val_X)
    #mean_absolute_error() function computes the mean absolute error between the predicted and actual salaries
    #If mae is low, the model is performing well
    val_mae = mean_absolute_error(val_predictions, val_y)
    print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

    # Model is trained again with a specific max_leaf_nodes value(leaf_nodes are the end points of the tree where predictions are made)
    #Limiting them makes the model simpler and can help prevent overfitting
    iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
    iowa_model.fit(train_X, train_y)
    val_predictions = iowa_model.predict(val_X)
    #evaluated with MAE again
    val_mae = mean_absolute_error(val_predictions, val_y)
    print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))
except FileNotFoundError:
    print("Error: The file was not found. Check the path and filename.")
except ValueError as ve:
    print("ValueError:", ve)
except Exception as e:
    print("An unexpected error occurred:", e)