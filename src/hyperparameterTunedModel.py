import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
import pickle

def train_model():
    # Load dataset
    df = pd.read_csv('data/house_data.csv')
    X = df[['bedrooms', 'bathrooms', 'floors', 'yr_built']]
    y = df['price']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Ridge Regression Model
    model = Ridge()

    # Set up the hyperparameter grid for tuning
    param_grid = {
        'alpha': [0.01, 0.1, 1, 10, 100]  # Regularization strength
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Best hyperparameters from grid search
    print(f"Best parameters found: {grid_search.best_params_}")

    # Get the best model from grid search
    best_model = grid_search.best_estimator_

    # Save the best model to a pickle file
    pickle.dump(best_model, open("hyperParameterTuned.pkl", "wb"))

    print("Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()
