import pickle
import pandas as pd

def test_model():
    with open('model/house_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Example test case
    new_house = pd.DataFrame([[3, 2.0, 1.0, 2000]], columns=['bedrooms', 'bathrooms', 'floors', 'yr_built'])
    prediction = model.predict(new_house)
    assert prediction[0] > 0, "Prediction should be a positive value."

if __name__ == "__main__":
    test_model()
