from data_preparation.generate_data import generate_time_series_data

df = generate_time_series_data(days=30)
print(df.head())

df.to_csv("sample_generated.csv", index=False)
print("Generated sample_generated.csv")
