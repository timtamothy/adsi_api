from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
app = FastAPI()
nn_pipe = load('../models/nn_pipe.joblib')

@app.get("/")
def read_root():
	return {"Hello": "AnyaChan"}

@app.get("/health", status_code=200)
def healthcheck():
	return "Anya is ready and very waku waku!"

def format_features(brewery: str, review_aroma: int, review_appearance: int, review_palate: int, review_taste:int):
	return{
		'Brewery': [brewery],
		'Aroma': [review_aroma],
		'Appearance': [review_appearance],
		'Palate': [review_palate]
		'Taste': [review_taste]
	}

@app.get("/beer_type")
def predict(brewery: str, review_aroma: int, review_appearance: int, review_palate: int, review_taste:int):
	features = format_features(brewery, review_aroma, review_appearance, review_palate, review_taste)
	obs = pd.DataFrame(features)
	pred = nn_pipe.predict(obs)
	return JSONResponse(pred.tolist())

