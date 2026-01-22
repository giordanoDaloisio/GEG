import sys
sys.path.append('..')


from argparse import ArgumentParser
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from utils import * 

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from fairlearn.reductions import ExponentiatedGradient, BoundedGroupLoss, ZeroOneLoss, GridSearch

import seaborn as sns

from demv import DEMV

sns.set_style('whitegrid')


def cross_val2(classifier, data, label, groups_condition, sensitive_features, positive_label, debiaser=None, exp=False, n_splits=10):
    fold = KFold(n_splits=n_splits, shuffle=True, random_state=2)
    metrics = {
        'stat_par' : [],
        'eq_odds': [],
        'zero_one_loss': [],
        'disp_imp': [],
        'acc': []
    }
    pred = None
    for train, test in fold.split(data):
        data = data.copy()
        df_train = data.iloc[train]
        df_test = data.iloc[test]
        model = deepcopy(classifier)
        if debiaser:
            print("?")
        else:
            run_metrics, predtemp = _model_train2(df_train, df_test, label, model, defaultdict(
                list), groups_condition, sensitive_features, positive_label, exp)
            pred = predtemp if pred is None else pred.append(predtemp)
        for k in metrics.keys():
            metrics[k].append(run_metrics[k])
    return model, metrics, pred

def _model_train2(df_train, df_test, label, classifier, metrics, groups_condition, sensitive_features, positive_label, exp=False):
    x_train, x_test, y_train, y_test = _train_test_split(
        df_train, df_test, label)
    model = deepcopy(classifier)
    model.fit(x_train, y_train,
              sensitive_features=df_train[sensitive_features]) if exp else model.fit(x_train, y_train)
    pred = model.predict(x_test)
    df_pred = df_test.copy()
    df_pred['y_true'] = df_pred[label]
    df_pred[label] = pred


    df_pred = blackbox(df_pred, label, sensitive_features[0], groups_condition)

    metrics['stat_par'].append(statistical_parity(  
        df_pred, groups_condition, label, positive_label))
    metrics['eq_odds'].append(equalized_odds(
        df_pred, groups_condition, label, positive_label))
    metrics['disp_imp'].append(disparate_impact(
        df_pred, groups_condition, label, positive_label=positive_label))
    metrics['zero_one_loss'].append(zero_one_loss_diff(
        y_true=y_test, y_pred=pred, sensitive_features=df_test[sensitive_features].values))
    metrics['acc'].append(accuracy_score(y_test, pred))
    return metrics, df_pred

def _train_test_split(df_train, df_test, label):
    x_train = df_train.drop(label, axis=1).values
    y_train = df_train[label].values.ravel()
    x_test = df_test.drop(label, axis=1).values
    y_test = df_test[label].values.ravel()
    return x_train, x_test, y_train, y_test


def blackbox(pred, label, sensitive_feat, unpriv_group):
    from balancers import MulticlassBalancer
    query = '&'.join([str(k) + '==' + str(v)
                     for k, v in unpriv_group.items()])
    label_query = label + '==' + str(positive_label)
    priv_group = pred.query('~(' + query + ')')
    priv_group_pos = pred.query('~(' + query + ')&' + label_query)

    for k,v in unpriv_group.items():
      key = k
      value = 1-v

    pb = MulticlassBalancer(y = 'y_true', y_ = label, a = sensitive_feat, data = pred)
    y_adj = pb.adjust(cv = True, summary = False)
    pred[label] = y_adj

    priv_group = pred.query('~(' + query + ')')
    priv_group_pos = pred.query('~(' + query + ')&' + label_query)

    if(len(priv_group_pos) == 0):
        unpriv_group_pos = pred.query('(' + query + ')&' + label_query).index
        pred.loc[unpriv_group_pos[0], key] = value

    return pred

pipeline = Pipeline([
  ('scaler', StandardScaler()),
  ('classifier', LogisticRegression())
])

#CMC

data = pd.read_csv('../data/cmc.data', names=['wife_age', 'wife_edu', 'hus_edu', 'num_child', 'wife_religion', 'wife_work', 'hus_occ', 'living', 'media', 'contr_use'])

label = 'contr_use'
sensitive_features = ['wife_work']
unpriv_group = { 'wife_work': 1}
positive_label= 2


model, blackboxmetrics, pred = cross_val2(pipeline, data, label, unpriv_group, sensitive_features, positive_label=positive_label)

save_metrics('blackbox_2', 'cmc', blackboxmetrics)


#LAW

data = pd.read_csv('../data/bar_pass_prediction.csv', index_col='Unnamed: 0')
col_to_drop = ['ID', 'decile1b', 'decile3', 'decile1', 'cluster', 'bar1', 'bar2',
               'sex', 'male', 'race1', 'race2', 'other', 'asian', 'black', 'hisp', 'bar', 'index6040', 'indxgrp', 'indxgrp2', 'dnn_bar_pass_prediction', 'grad', 'bar1_yr', 'bar2_yr', 'ugpa']
data.drop(col_to_drop, axis=1, inplace=True)
data.loc[data['Dropout'] == 'NO', 'Dropout'] = 0
data.loc[data['Dropout'] == 'YES', 'Dropout'] = 1
data.dropna(inplace=True)
data.loc[data['gender']=='female', 'gender'] = 1
data.loc[data['gender'] == 'male', 'gender'] = 0
data.loc[data['race']==7.0, 'race'] = 0
data.loc[data['race'] != 0, 'race'] = 1
data['gpa'] = pd.qcut(data['gpa'], 3, labels=['a','b','c'])
enc = LabelEncoder()
data['gpa'] = enc.fit_transform(data['gpa'].values)

protected_group = {'gender': 1}
label = 'gpa'
sensitive_features=['gender']
positive_label = 2

model, blackboxmetrics, pred = cross_val2(pipeline, data, label, protected_group, sensitive_features, positive_label=positive_label)

save_metrics('blackbox_2', 'law', blackboxmetrics)


#TRUMP

def make_data():
    data = pd.read_csv('../data/data_e28.csv', index_col='[meta] uuid')
    data.rename(columns = lambda c: c[c.find("]")+1:].replace("_", " ").upper().strip(), inplace=True)
    
    voted = data['VOTED PARTY LAST ELECTION DE'][data['COUNTRY CODE'] == 'DE']\
    .append(data['VOTED PARTY LAST ELECTION IT'][data['COUNTRY CODE'] == 'IT'])\
    .append(data['VOTED PARTY LAST ELECTION FR'][data['COUNTRY CODE'] == 'FR'])\
    .append(data['VOTED PARTY LAST ELECTION GB'][data['COUNTRY CODE'] == 'GB'])\
    .append(data['VOTED PARTY LAST ELECTION ES'][data['COUNTRY CODE'] == 'ES'])\
    .append(data['VOTED PARTY LAST ELECTION PL'][data['COUNTRY CODE'] == 'PL'])

    rankingParty = data['RANKING PARTY DE'][data['COUNTRY CODE'] == 'DE']\
    .append(data['RANKING PARTY IT'][data['COUNTRY CODE'] == 'IT'])\
    .append(data['RANKING PARTY FR'][data['COUNTRY CODE'] == 'FR'])\
    .append(data['RANKING PARTY GB'][data['COUNTRY CODE'] == 'GB'])\
    .append(data['RANKING PARTY ES'][data['COUNTRY CODE'] == 'ES'])\
    .append(data['RANKING PARTY PL'][data['COUNTRY CODE'] == 'PL'])

    voteNextElection = pd.concat([data['VOTE NEXTELECTION DE'][data['COUNTRY CODE'] == 'DE'],
                                    data['VOTE NEXTELECTION IT'][data['COUNTRY CODE'] == 'IT'],
                                    data['VOTE NEXTELECTION FR'][data['COUNTRY CODE'] == 'FR'],
                                    data['VOTE NEXTELECTION GB'][data['COUNTRY CODE'] == 'GB'],
                                    data['VOTE NEXTELECTION ES'][data['COUNTRY CODE'] == 'ES'],
                                    data['VOTE NEXTELECTION PL'][data['COUNTRY CODE'] == 'PL']], verify_integrity=True)

    data['VOTED PARTY LAST ELECTION'] = voted
    data['RANKING PARTY'] = rankingParty
    data['VOTE NEXT ELECTION'] = voteNextElection

    data.drop(['VOTED PARTY LAST ELECTION DE', 'VOTED PARTY LAST ELECTION IT', 'VOTED PARTY LAST ELECTION FR',
               'VOTED PARTY LAST ELECTION GB', 'VOTED PARTY LAST ELECTION ES', 'VOTED PARTY LAST ELECTION PL',
               'RANKING PARTY DE', 'RANKING PARTY IT', 'RANKING PARTY FR', 'RANKING PARTY GB', 'RANKING PARTY ES', 
               'RANKING PARTY PL', 'VOTE NEXTELECTION DE', 'VOTE NEXTELECTION IT', 'VOTE NEXTELECTION FR', 'VOTE NEXTELECTION GB',
               'VOTE NEXTELECTION ES', 'VOTE NEXTELECTION PL'], axis=1, inplace=True)

    data.drop('VOTE REFERENDUM', axis = 1, inplace=True)

    data.drop('EMPLOYMENT STATUS IN EDUCATION', axis=1, inplace=True)

    data.drop('ORIGIN', axis=1, inplace=True)

    data['MEMBER ORGANIZATION'].fillna('Not member', inplace=True)
    data.loc[data['MEMBER ORGANIZATION']=='Not member', 'ORGANIZATION ACTIVITIES TIMEPERWEEK'] = 'Not member'

    data.drop(data.loc[data['HOUSEHOLD SIZE'].isnull()].index, inplace=True)

    data.drop(data.loc[data['SOCIAL NETWORKS REGULARLY USED'].isnull()].index, inplace=True)

    nullcols = data.isna().any()[data.isna().any()==True].index
    data.drop(nullcols, axis=1, inplace=True)

    data.drop('WEIGHT', axis=1, inplace=True)
    data.loc[data['GENDER']=='male', 'GENDER'] = 1
    data.loc[data['GENDER']!=1, 'GENDER'] = 0
    data['GENDER'] = data['GENDER'].astype(int)
    data.loc[data['RELIGION'] == 'Roman Catholic', 'RELIGION'] = 1
    data.loc[data['RELIGION'] != 1, 'RELIGION'] = 0
    data['RELIGION'] = data['RELIGION'].astype(int)
    enc = LabelEncoder()
    data['POLITICAL VIEW'] = enc.fit_transform(data['POLITICAL VIEW'].values)
    data.rename(columns= lambda c: c.replace(" ", "_"), inplace=True)
    for c in data.columns:
        if len(data[c].unique())>6:
            data.drop(c, axis=1, inplace=True)
    return data

data = make_data()

data = pd.get_dummies(data)

label = 'POLITICAL_VIEW'
protected_group = {'RELIGION': 0}
sensitive_variables=['RELIGION']
positive_label = 3

model, blackboxmetrics, pred = cross_val2(pipeline, data, label, protected_group, sensitive_features=sensitive_variables, positive_label=positive_label)

save_metrics('blackbox_2', 'trump', blackboxmetrics)


#CRIME

def prepare_data():
  
  data = pd.read_excel('../data/crime_data_normalized.xlsx', na_values='?')
  data.drop(['state', 'county', 'community', 'communityname',
            'fold', 'OtherPerCap'], axis=1, inplace=True)
  na_cols = data.isna().any()[data.isna().any() == True].index
  data.drop(na_cols, axis=1, inplace=True)
  data = (data - data.mean())/data.std()
  y_classes = np.quantile(data['ViolentCrimesPerPop'].values, [
                          0, 0.2, 0.4, 0.6, 0.8, 1])
  i = 0
  data['ViolentCrimesClass'] = data['ViolentCrimesPerPop']
  for cl in y_classes:
    data.loc[data['ViolentCrimesClass'] <= cl, 'ViolentCrimesClass'] = i*100
    i += 1
  data.drop('ViolentCrimesPerPop', axis=1, inplace=True)
  data['black_people'] = data['racepctblack'] > -0.45
  data['hisp_people'] = data['racePctHisp'] > -0.4
  data['black_people'] = data['black_people'].astype(int)
  data['hisp_people'] = data['hisp_people'].astype(int)
  data.drop('racepctblack', axis=1, inplace=True)
  data.drop('racePctHisp', axis=1, inplace=True)
  return data

data = prepare_data()

label = 'ViolentCrimesClass'
groups_condition = {'hisp_people': 1}
sensitive_features = ['hisp_people']
positive_label = 100

model, blackboxmetrics, pred = cross_val2(pipeline, data, label, groups_condition, sensitive_features, positive_label=positive_label)

save_metrics('blackbox_2', 'crime', blackboxmetrics)


#PARK

def prepare_data():
  data = pd.read_csv('../data/park.csv')
  data.drop(['subject#', 'a', 'y', 'yhat', 'motor_UPDRS', 'total_UPDRS', 'test_time'], axis=1, inplace=True)
  data.loc[data['age']<65, 'age'] = 0
  data.loc[data['age']>=65, 'age'] = 1
  data['score_cut'].replace({
    'Mild': 0,
    'Moderate': 1,
    'Severe': 2
  }, inplace=True)
  changed_labels = data[(data['age']==1)&(data['sex']==1)&(data['score_cut']==1)].sample(n=200).index
  data.loc[changed_labels, 'score_cut'] = 0
  return data

data = prepare_data()

label = 'score_cut'
sensitive_vars = ['sex']
protected_group = {'sex': 0}
positive_label = 0

model, blackboxmetrics, pred = cross_val2(pipeline, data, label, protected_group, sensitive_vars, positive_label)

save_metrics('blackbox_2', 'park', blackboxmetrics)


#OBESITY

def prepare_data():
  data = pd.read_csv('../data/obesity.csv')
  data.drop(['NObeyesdad', 'weight_cat', 'yhat', 'a'], axis=1, inplace=True)
  le = LabelEncoder()
  data['Gender'] = le.fit_transform(data['Gender'].values)
  data['y'].replace({
    'Normal_Weight': 0,
    'Overweight_Level_I': 1,
    'Overweight_Level_II': 2,
    'Obesity_Type_I': 3,
    'Insufficient_Weight': 4
  }, inplace=True)
  data['family_history_with_overweight']=le.fit_transform(data['family_history_with_overweight'].values)
  data['FAVC'] = le.fit_transform(data['FAVC'].values)
  data['CAEC'] = le.fit_transform(data['CAEC'].values)
  data['SMOKE'] = le.fit_transform(data['SMOKE'].values)
  data['SCC'] = le.fit_transform(data['SCC'].values)
  data['CALC'] = le.fit_transform(data['CALC'].values)
  data['MTRANS'] = le.fit_transform(data['MTRANS'].values)
  data.loc[data['Age'] < 22 , 'Age'] = 0
  data.loc[data['Age'] >= 22, 'Age'] = 1
  return data


data = prepare_data()

data = data.loc[data['y'] != 4]

label = 'y'
positive_label = 0
protected_group = {'Age': 1}
sensitive_vars = ['Age']

data = data.sample(frac=1).reset_index(drop=True)

model, blackboxmetrics, pred = cross_val2(pipeline, data, label, protected_group, sensitive_vars, positive_label)

save_metrics('blackbox_2', 'obesity', blackboxmetrics)


#WINE

def load_data():
  red = pd.read_csv('../data/winequality-red.csv', sep=';')
  red['type'] = 0
  white = pd.read_csv('../data/winequality-white.csv', sep=';')
  white['type'] = 1
  data = red.append(white)
  data.drop(data[(data['quality']==3)|(data['quality']==9)].index, inplace=True)
  data.loc[data['alcohol'] <= 10, 'alcohol'] = 0
  data.loc[(data['alcohol'] > 10) & (data['alcohol'] != 0), 'alcohol'] = 1
  return data

data = load_data()

data = data.loc[ data.quality != 8 ]

label = 'quality'
sensitive_variables = ['type']
protected_group = {'type': 1}
positive_label = 6

data = data.sample(frac=1).reset_index(drop=True)
data = data.sample(frac=1).reset_index(drop=True)
data = data.sample(frac=1).reset_index(drop=True)


model, blackboxmetrics, pred = cross_val2(pipeline, data, label, protected_group, sensitive_variables, positive_label)

save_metrics('blackbox_2', 'wine', blackboxmetrics)


#DRUG

def prepare_data():
  data = pd.read_csv('../data/drugs.csv')
  data.drop(['yhat','a'], axis=1, inplace=True)
  data.loc[data['gender']==0.48246,'gender']=1
  data.loc[data['gender']==-0.48246,'gender']=0
  data['y'].replace({
    'never': 0,
    'not last year': 1,
    'last year': 2}, inplace=True)
  data['race'].replace({
    'non-white': 0,
    'white': 1}, inplace=True)
  string_cols = data.dtypes[data.dtypes == 'object'].index.values
  data.drop(string_cols, axis=1, inplace=True)
  return data


data = prepare_data()

label = 'y'
protected_group = {'race':1 }
positive_label = 0
sensitive_features = ['race']


model, blackboxmetrics, pred = cross_val2(pipeline, data, label, protected_group, sensitive_features, positive_label)

save_metrics('blackbox_2', 'drug', blackboxmetrics)

print([len(ids) for ids in group_ids])
print([tools.p_vec(y_[ids]).shape for ids in group_ids])