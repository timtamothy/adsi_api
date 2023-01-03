from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
app = FastAPI()
nn_pipe2 = load('../models/models/nn_pipe2.joblib')

@app.get('/')
def read_root():
	return {'This API is designed to return a prediction of beer type based on the following inputs: 1. Brewery Name, 2. Aroma Score, 3. Appearance Score, 4. Palate Score, 5. Taste Score, 6. Beer Alcohol by Volume. The endpoints of the API are /beer/type/ (GET) for single beer predictions or /beers/type/ (GET) for multiple beer predictions. A /health/ (GET) endpoint is also provided to display API status.'}

@app.get('/health', status_code=200)
def healthcheck():
	return 'The API is running poifectly'

def format_features(brewery: str, review_aroma: int, review_appearance: int, review_palate: int, review_taste: int):
	return {
		'brewery_name': [brewery],
		'Aroma': [review_aroma],
		'Appearance': [review_appearance],
		'Palate': [review_palate],
		'Taste': [review_taste],
		'ABV': [beer_abv]
	}

@app.get("/beer/type")
def predict(brewery: str, review_aroma: int, review_appearance: int, review_palate: int, review_taste:int, beer_abv:int):
	features = format_features(brewery, review_aroma, review_appearance, review_palate, review_taste, beer_abv)
	obs = pd.DataFrame(features)
	pred = nn_pipe2.predict(obs)
	return JSONResponse(pred.tolist())

@app.get("/beers/type")
def predict(brewery: str, review_aroma: int, review_appearance: int, review_palate: int, review_taste:int, beer_abv:int):
	features = format_features(brewery, review_aroma, review_appearance, review_palate, review_taste, beer_abv)
	obs = pd.DataFrame(features)
	pred = nn_pipe2.predict(obs)
	return JSONResponse(pred.tolist())

#pydantic basemodel