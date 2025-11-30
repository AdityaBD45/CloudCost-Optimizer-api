from data_preparation.generate_data import generate_time_series_data
from ml.cost_predictor import CostPredictor

df = generate_time_series_data(days=30)   # more data = better model

model = CostPredictor()
result = model.analyze(df)

print(result)

