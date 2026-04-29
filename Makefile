# Makefile for Credit Risk Assessment

.PHONY: install clean data clean-data run-all

install:
	pip install -r requirements.txt

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-data:
	rm -rf data/processed/*.csv
	rm -rf models/saved_models/*.pkl
	rm -rf reports/*.png

run-all:
	python 01_Data_Loading_Checking.py
	python 02_Data_Cleaning.py
	python 03_EDA_Visualization.py
	python 04_Feature_Engineering.py
	python 05_Modeling.py

test:
	python -m pytest tests/