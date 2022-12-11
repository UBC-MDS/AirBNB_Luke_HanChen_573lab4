# customer complaint analysis
# author: Ty Andrews, Dhruvi Nishar, Luke Yang
# date: 2022-11-30

all: results/eda_results/categorical_result.png results/models/feature_Importance.csv results/final_report.html

# pre-process data (e.g., scale and split into train & test)
data/processed/test_cleaned.csv data/processed/train_cleaned.csv : data/raw/AB_NYC_2019.csv
	python src/data_preprocessor.py --input="data/raw/AB_NYC_2019.csv" --output="data/processed/" 

# exploratory data analysis - visualize predictor distributions across classes
results/eda_results/categorical_result.png results/eda_results/lastreview_result.png: src/data_eda.py data/processed/train_cleaned.csv
	python src/data_eda.py --traindata=data/processed/train_cleaned.csv --output=results/eda_results/

# perform analysis 
results/models/feature_Importance.csv results/models/model_performance.png: src/data_analysis.py data/processed/test_cleaned.csv data/processed/train_cleaned.csv
	python src/data_analysis.py --traindata=data/processed/train_cleaned.csv --testdata=data/processed/test_cleaned.csv --output=results/models/

results/final_report.html: lab4.ipynb
	jupyter nbconvert --to html lab4.ipynb

clean: 
	rm -f data/**/*.csv
	rm -f results/**/*.png
	rm -f results/**/*.csv