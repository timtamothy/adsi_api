from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
app = FastAPI()
nn_pipe2 = load('../models/models/nn_pipe2.joblib')

@app.get('/')
def read_root():
	return {'Hello': 'This is Anya-Chan'}

@app.get('/health', status_code=200)
def healthcheck():
	return 'Anya is ready and very waku waku! Angus is ready and very wagyu wagyu'

def format_features(brewery: str, review_aroma: int, review_appearance: int, review_palate: int, review_taste: int):
	return {
		'brewery_name': [brewery],
		'Aroma': [review_aroma],
		'Appearance': [review_appearance],
		'Palate': [review_palate],
		'Taste': [review_taste]
	}

@app.get("/beer_type")
def predict(brewery: str, review_aroma: int, review_appearance: int, review_palate: int, review_taste:int):
	features = format_features(brewery, review_aroma, review_appearance, review_palate, review_taste)
	obs = pd.DataFrame(features)
	pred = nn_pipe2.predict(obs)
	return JSONResponse(pred.tolist())

