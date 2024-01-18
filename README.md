# End To End Machine Learning
This project is written based on getting data from a CSV file, however it can be changed to read from a database by changing line 30 of data_ingestion.py file.
The process has 3 steps:
  1.  load the data
  2.  clean and transform the data and prepare it to create a model
  3.  train different models and find the best performing model and its parameters using GridSearchCV algorithm
  4.  create a pickle file of the model to be used easily in the future for prediction.
# How to run
1.  create a conda venv and activate it:
    conda create .venv myenv python={python version}
    conda activate .venv/
2.  run the following:
    pip install -r requirements.txt  => It is to install all requirements
    python src/components/data_ingestion.py
