params={
    "Random Forest":{
    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    
    'max_features':['sqrt','log2',None],
    'n_estimators': [8,16,32,64,128,256]
    },
    "Decision Tree": {
        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'splitter':['best','random'],
        'max_features':['sqrt','log2'],
    },
    "Gradient Boosting":{
        'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
        'learning_rate':[.1,.05,.01],
        # 'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
        'criterion':['squared_error', 'friedman_mse'],
        # 'max_features':['sqrt','log2'],
        # 'n_estimators': [8,16,32,64,128,256]
    },
    "Linear Regression":{

    },
    "AdaBoost Regressor":{
        'learning_rate':[.1,.05,.01],
        'loss':['linear','square','exponential'],
        'n_estimators': [8,16,32,64,128,256]
    },
    "K-Neighbors Regressor":{
        'n_neighbors':[3, 5, 8, 10],
        'weights':['uniform', 'distance'],
        'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
    }
}