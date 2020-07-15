import pandas as pd
import numpy as np
from ipywidgets import Output
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
import itertools  
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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
from IPython.core.display import display as display_core, HTML
from IPython.display import display as dis
import ipywidgets as widgets
from IPython.display import clear_output


def model_selection(X_train, X_test, y_train, y_test):
    warnings.filterwarnings('ignore')
    toggle = widgets.ToggleButtons(
                    options=['Fast', 'Regular', 'Slow'],
                    description='Speed:',
                    index = None,
                    disabled=False,
                    button_style='',
                    tooltips=['Without hyper-parameters optimization', 'Fast hyper-parameters optimization', 'Full hyper-parameters optimization'])
    toggle.observe(build_models, names=['value'])
    display_core(toggle)

def build_models(Speed):
    global df_metrics, models_list, list_model_name
    
    list_model_name = ['Logistic Regression',
                       'KNN',
                       'Decision Tree',
                       'SGD',
                       'Random Forest',
                       'Gradient Boosting',
                       'XGBoost',
                       'LGBM',
                       'CatBoost']
    
    display.clear_output(wait=True)
    prediction_type = get_prediction_type(y_train)
    if prediction_type == 'multiclass':
            list_metrics = ['Accuracy', 'F1', '1 - log loss', '1 - MSE']
    else:
            list_metrics = ['Accuracy', 'AUC', 'F1', '1 - log loss', '1 - MSE']
    models_list = create_list_models(X_train, y_train)
    df_metrics = pd.DataFrame(columns=list_metrics)
    if Speed['new'] == 'Fast':
        fast_model_selection(df_metrics, prediction_type, X_test, y_test)
    if Speed['new'] == 'Regular':
        regular_model_selection(df_metrics)
    if Speed['new'] == 'Slow':
        slow_model_selection(df_metrics)
        

def fast_model_selection(df_metrics, prediction_type, X_test, y_test):
        global confusion_matrix_dict
        out = Output()
        display.display(out)
        confusion_matrix_dict = {}
        for model in tqdm(models_list):
            
            model[0].fit(X_train, y_train)

            df_metrics = create_df_metrics(model, df_metrics, prediction_type, X_test, y_test)

            with out:
                html = (df_metrics.style
                          .apply(highlight_min)
                          .apply(highlight_max))
                display.clear_output(wait=True)
                display.display(html)
                confusion_matrix_dict[model[1]] = [model[0], y_test, model[0].predict(X_test)]

        make_confusion_matrix()         
            
def regular_model_selection(df_metrics):
    print('to do fast optimizotion')
    
def slow_model_selection(df_metrics):
    print('to do full optimizotion')

def make_confusion_matrix():

    def on_button_clicked(selection):
        with out:
            clear_output(wait=True)
            x = plot_confusion_matrix(selection['new'])
            x.show()

    button = widgets.ToggleButtons(
            options = list_model_name,
            description='CM:',
            disabled=False,
            button_style='')
    button.observe(on_button_clicked, names=['value'])
    dis(button)
    out = Output()
    dis(out)        
            
def create_df_metrics(model, df_metrics, prediction_type, X_test, y_true):
    
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
    
def plot_confusion_matrix(model_name,  cmap=plt.cm.Blues, normalize=False):
    
    model = confusion_matrix_dict[model_name][0]
    y_true = confusion_matrix_dict[model_name][1]
    y_pred = confusion_matrix_dict[model_name][2]
    cm = confusion_matrix(y_true, y_pred)
    classes = model.classes_
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(model_name)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt
