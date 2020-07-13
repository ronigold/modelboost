def model_selection(X_train, X_test, y_train, y_test):
    
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import time
    from IPython import display
    from tqdm import tqdm
    
    from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import SGDClassifier
    
    list_metrics = ['accuracy', 'auc']
    seed = 42
    models_list = []
    lr = LogisticRegression(random_state=seed)
    models_list.append([lr, 'Logistic Regression'])
    knn = KNeighborsClassifier()
    models_list.append([knn, 'KNN'])
    dt = DecisionTreeClassifier(random_state=seed)
    models_list.append([dt, 'Decision Tree'])
    svm = SGDClassifier(max_iter=1000, tol=0.001, random_state=seed)
    models_list.append([svm, 'SGD'])
    rf = RandomForestClassifier(n_estimators=10, random_state=seed)
    models_list.append([rf, 'Random Forest'])
    gbc = GradientBoostingClassifier(random_state=seed)
    models_list.append([gbc, 'Gradient Boosting'])
    xgboost = XGBClassifier(random_state=seed, n_jobs=-1, verbosity=0)
    models_list.append([xgboost, 'XGBoost'])
    lightgbm = LGBMClassifier(random_state=seed)
    models_list.append([lightgbm, 'LGBM'])
    catboost = CatBoostClassifier(random_state=seed, silent = True) 
    models_list.append([catboost, 'CatBoost'])
    df_metrics = pd.DataFrame(columns=list_metrics)
    
    for model in tqdm(models_list):
        model[0].fit(X_train, y_train)
        predict = model[0].predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predict)
        auc = metrics.roc_auc_score(y_test, predict)
        df_metrics.loc[model[1]] = {'accuracy': accuracy, 'auc': auc}
        display.clear_output(wait=True)
        display.display(df_metrics)
        
    return df_metrics

