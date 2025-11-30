from data_preparation.parse_input import parse_input

df = parse_input("sample_generated.csv", is_csv=True)
print(df.head())
