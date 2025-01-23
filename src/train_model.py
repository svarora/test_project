import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


def train_model():
    df = pd.read_csv('data/house_data.csv')
    X = df[['bedrooms', 'bathrooms', 'floors', 'yr_built']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)    
    model = LinearRegression()
    model.fit(X_train, y_train)
    with open('model/house_price_model.pkl', 'wb') as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    train_model()
