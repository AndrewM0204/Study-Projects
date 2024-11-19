import pickle
import pandas as pd

def house_price_model1(X: pd.DataFrame):
	with open("model1.pkl", "rb") as f:
		model1 = pickle.load(f)
	return model1.predict(X)
