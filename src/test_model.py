import pickle
import pandas as pd


def test_model():
    with open('model/house_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
    new_house_data = [[3, 2.0, 1.0, 2000]]
    new_house_df = pd.DataFrame(new_house_data, 
                                columns=['bedrooms', 'bathrooms', 'floors', 'yr_built'])
    predicted_price = model.predict(new_house_df)
    print(f"Predicted price for the new house: ${predicted_price[0]:,.2f}")


if __name__ == "__main__":
    test_model()

    