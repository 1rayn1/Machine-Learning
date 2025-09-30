import pandas as pd #Loading and Manipulating dataset
import os #for making sure file paths work and exist
import joblib # for saving and loading models
from sklearn.linear_model import SGDRegressor #used to create the model (Regressor type)
from sklearn.preprocessing import StandardScaler #for normalizing features to have mean 0 and variance 1
from sklearn.metrics import mean_absolute_error #used to measure model performance
from sklearn.model_selection import train_test_split #used to split data into training and validation sets

# Paths
file_path = 'car data.csv' #if the csv file changes, update this path
model_file = 'sgd_car_model.pkl' 
scaler_file = 'scaler.pkl'

while True:
    reset_model = True

    inputing = input("Reset data? (y/n): ")
    if inputing.lower() == "y":
        reset_model = True
    elif inputing.lower() == "n":
        reset_model = False
    elif inputing.lower() == "quit" or inputing.lower() == "q":
        print("Exiting program.")
        break
    try:
        # Load dataset
        # Reset flag: set to True to restart model and scaler from scratch

        # Delete saved model and scaler if reset is requested
        if reset_model:
            if os.path.exists(model_file):
                os.remove(model_file)
                print("Deleted saved model.")
            if os.path.exists(scaler_file):
                os.remove(scaler_file)
                print("Deleted saved scaler.")
            data = pd.read_csv(file_path, index_col=0)

        # Target and features

        y = data['Present_Price'].values # target variable 
        #if the csv file changes, update this line

        features = ['Year', 'Selling_Price', 'Kms_Driven'] # features to use to predict target
        #if the csv file changes, update this line

        X = data[features].values # feature matrix
        #if the csv file changes, update this line

        # Split for evaluation
        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1) # 75% train, 25% validation

        #Load or create scaler for feature normalization
        #checks if scaler file exists
        #if it does, load it
        #if not, create it and save it
        if os.path.exists(scaler_file):
            scaler = joblib.load(scaler_file) # load existing scaler
        else:
            scaler = StandardScaler()
            scaler.fit(train_X)
            joblib.dump(scaler, scaler_file)

        train_X = scaler.transform(train_X) #apply normalization to training data
        val_X = scaler.transform(val_X) #apply normalization to validation data
    
        # Load or create model
        if os.path.exists(model_file):
            print("Loading saved incremental model...")
            model = joblib.load(model_file)
        else:
            print("No saved model found. Creating new incremental model...")
            model = SGDRegressor(
                max_iter=1,   # one epoch per partial_fit
                learning_rate='invscaling',
                eta0=0.01,
                warm_start=True,
                random_state=1
            )

        # Partial fit (incremental learning)
        #performs one round of training on the training data
        model.partial_fit(train_X, train_y)

        # Evaluate
        #predicts the target values for the validation set
        #calculates the mean absolute error (MAE) between the predicted and actual target values

        preds = model.predict(val_X)
        mae = mean_absolute_error(val_y, preds)
        print(f"MAE after incremental training: {mae:.2f}")


        # Save model
        joblib.dump(model, model_file)
        print("Incremental model saved.")


    except Exception as e:
        print("Error:", e)