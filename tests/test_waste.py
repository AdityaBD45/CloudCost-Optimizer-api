from data_preparation.generate_data import generate_time_series_data
from ml.waste_detector import run_waste_detection

df = generate_time_series_data(days=7)

result = run_waste_detection(df)

print(result)
