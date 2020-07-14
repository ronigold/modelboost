def model_selection(X_train, X_test, y_train, y_test):
    
    from ipywidgets import Output
    import warnings
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    import time
    from IPython import display
    from tqdm.notebook import trange, tqdm
    
    from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.calibration import CalibratedClassifierCV
    
    warnings.filterwarnings('ignore')

    prediction_type = get_prediction_type(y_train)
    if prediction_type == 'multiclass':
            list_metrics = ['Accuracy', 'F1', '1 - log loss', '1 - MSE']
    else:
            list_metrics = ['Accuracy', 'AUC', 'F1', '1 - log loss', '1 - MSE']
    models_list = create_list_models(X_train, y_train)
    df_metrics = pd.DataFrame(columns=list_metrics)
    out = Output()
    display.display(out)
    
    for model in tqdm(models_list):
        model[0].fit(X_train, y_train)
        
        df_metrics = create_df_netrics(model, df_metrics, prediction_type, y_test)
            
        with out:
            html = (df_metrics.style
                      .apply(highlight_min)
                      .apply(highlight_max))
            display.clear_output(wait=True)
            display.display(html)
        
    return df_metrics

def create_df_netrics(model, df_metrics, prediction_type, y_true):
    
    y_pred = model[0].predict(X_test)
    pred_proba = model[0].predict_proba(X_test)
    
    accuracy = metrics.accuracy_score(y_true, y_pred)
    
    log_loss = metrics.log_loss(y_true,pred_proba)
    log_loss = 1 - log_loss
    
    mse = metrics.mean_squared_error(y_true, y_pred)
    mse = 1 - mse
    if prediction_type == 'multiclass': 
            f1 = metrics.f1_score(y_true, y_pred, average='macro') 
            df_metrics.loc[model[1]] = {'Accuracy': accuracy, 'F1': f1, '1 - log loss': log_loss, '1 - MSE': mse}
            
    else:
        f1 = metrics.f1_score(y_true, y_pred) 
        auc = metrics.roc_auc_score(y_true, y_pred)
        df_metrics.loc[model[1]] = {'Accuracy': accuracy, 'AUC': auc, 'F1': f1, '1 - log loss': log_loss, '1 - MSE': mse}
        
    return df_metrics
        
def create_list_models(X_train, y_train, seed = 42):
    models_list = []
    lr = LogisticRegression(random_state=seed)
    models_list.append([lr, 'Logistic Regression'])
    knn = KNeighborsClassifier()
    models_list.append([knn, 'KNN'])
    dt = DecisionTreeClassifier(random_state=seed)
    models_list.append([dt, 'Decision Tree'])
    base_svm = SGDClassifier(max_iter=1000, tol=0.001, loss = 'hinge', random_state=seed)
    base_svm.fit(X_train, y_train)
    svm = CalibratedClassifierCV(base_svm,  cv='prefit')
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
    return models_list

def get_prediction_type(y_train):
        prediction_type = 'multiclass'
        if len(y_train.value_counts()) == 2:
            prediction_type = 'binary'
        return prediction_type

def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),
                            index=data.index, columns=data.columns)

def highlight_min(data, color='red'):
    '''
    highlight the minimum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_min = data == data.min()
        return [attr if v else '' for v in is_min]
    else:  # from .apply(axis=None)
        is_min = data == data.min().min()
        return pd.DataFrame(np.where(is_min, attr, ''),
                            index=data.index, columns=data.columns)