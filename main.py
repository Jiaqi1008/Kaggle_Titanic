import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier as ada
from sklearn import naive_bayes, svm, tree
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
import re


def set_missing_ages(df):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    known_age = age_df[age_df.Age.notnull()]
    unknown_age = age_df[age_df.Age.isnull()]

    y = known_age.iloc[:, 0]
    X = known_age.iloc[:, 1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    predictedAges = rfr.predict(unknown_age.iloc[:, 1:])
    df.loc[(df.Age.isnull()), 'Age'] = np.rint(predictedAges)
    return df, rfr

def set_age_type(df):
    age_listbins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 200]
    age_listlabel = ['0_5', '6_10', '11_15', '16_20', '21_25', '26_30', '31_35', '36_40', '41_45',
                     '46_50', '51_55', '56_60', '61_65', '66_70', '70+']
    df['Age_bucket'] = pd.cut(data_train.Age, age_listbins, labels=age_listlabel)
    return df

def set_SibSp_type(df):
    SibSp_listbins = [-1, 0, 1, 2, 3, 4, 20]
    SibSp_listlabel = ['0', '1', '2', '3', '4', '4+']
    df['SibSp_bucket'] = pd.cut(data_train.SibSp, SibSp_listbins, labels=SibSp_listlabel)
    return df

def set_Parch_type(df):
    Parch_listbins = [-1, 0, 1, 2, 20]
    Parch_listlabel = ['0', '1', '2', '2+']
    df['Parch_bucket'] = pd.cut(data_train.Parch, Parch_listbins, labels=Parch_listlabel)
    return df

def set_family_size_type(df):
    family_size_bin = [0, 1, 4, 20]
    family_size_binlabel = ['Single', 'Small', 'Large']
    df['family'] = data_train.SibSp + data_train.Parch + 1
    df['Family_size_category'] = pd.cut(data_train.family,family_size_bin, labels=family_size_binlabel)

def set_Fare_bucket(df):
    fare_bins = [-1, 10, 50, 100, 600]
    fare_binlabel = ['0_10', '11_50', '51_100', '100+']
    df['Fare_bucket'] = pd.cut(data_train.Fare, fare_bins, labels=fare_binlabel)
    return df

def set_Cabin_type(df):
    df.loc[(df['Cabin'].isna()), 'Cabin'] = '0'
    df['Cabin'] = df['Cabin'].map(lambda x: x.split()[-1])
    df['Cabin_class'] = df['Cabin'].map(lambda x: re.compile("[A-Z]").findall(x))
    df['Cabin_class'] = df['Cabin_class'].map(lambda x: str(x)[2:-2])
    df.loc[(~(df['Cabin'] == '0')), 'Cabin'] = '1'
    return df

def set_MorC_type(df):
    df['Mother'] = ''
    df.loc[(data_train['Parch'] > 0) & (data_train['Title'] == 'Mrs'), 'Mother'] = 1
    df.loc[(data_train['Parch'] == 0) | (data_train['Title'] != 'Mrs'), 'Mother'] = 0
    df['Child'] = ''
    df.loc[(data_train['Age'] <= 12), 'Child'] = 1
    df.loc[(data_train['Age'] > 12), 'Child'] = 0

def set_title_type(df):
    title_Dict = {}
    title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
    title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
    title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
    title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
    title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
    title_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
    df['Title_type'] = df['Title'].map(title_Dict)

if __name__ =='__main__':
    data_train = pd.read_csv("train.csv")
    # rebuild the missing data
    data_train, rfr = set_missing_ages(data_train)
    # data washing
    data_train['Title'] = data_train['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
    set_age_type(data_train)
    set_SibSp_type(data_train)
    set_Parch_type(data_train)
    set_Cabin_type(data_train)
    set_family_size_type(data_train)
    set_Fare_bucket(data_train)
    set_title_type(data_train)
    set_MorC_type(data_train)

    dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
    dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')
    dummies_Age = pd.get_dummies(data_train['Age_bucket'], prefix='Age_bucket')
    dummies_SibSp = pd.get_dummies(data_train['SibSp_bucket'], prefix='SibSp')
    dummies_Parch = pd.get_dummies(data_train['Parch_bucket'], prefix='Parch')
    dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
    dummies_Cabin=pd.get_dummies(data_train['Cabin'],prefix='Cabin')
    dummies_Family_size = pd.get_dummies(data_train['Family_size_category'], prefix='Family_size_category')
    dummies_Fare=pd.get_dummies(data_train['Fare_bucket'],prefix='Fare_bucket')
    dummies_Title= pd.get_dummies(data_train['Title_type'], prefix='Title')
    dummies_Cabin_class = pd.get_dummies(data_train['Cabin_class'], prefix='Cabin_class')

    train_df = pd.concat([data_train, dummies_Sex, dummies_Pclass,dummies_Age,dummies_SibSp,dummies_Parch,
                    dummies_Embarked,dummies_Cabin,dummies_Family_size,dummies_Fare,dummies_Title,
                    dummies_Cabin_class], axis=1)
    col=data_train.columns.values
    # normalization
    scaler = preprocessing.StandardScaler()
    train_df['Age_scaled'] = scaler.fit_transform(train_df[['Age']])
    train_df['Fare_scaled'] = scaler.fit_transform(train_df[['Fare']])
    train_df.drop(col[2:-2], axis=1, inplace=True)
    train_df.drop(['SibSp_0','Parch_0','Cabin_class_T'], axis=1, inplace=True)
    # set a model
    train_np = train_df.values
    train_df.to_csv("train_washed.csv", index=False)
    y = train_np[:, 1]
    x = train_np[:, 2:]
    # LR
    # model_name='logistic_regression'
    # clf=linear_model.LogisticRegression(C=0.14, penalty='l2', tol=1e-6)
    # ada
    # model_name='ada_boost'
    # clf=ada(linear_model.LogisticRegression(C=1, penalty='l2', tol=1e-6), algorithm='SAMME.R',
    #           learning_rate=0.7, n_estimators=65, random_state=7)
    # SVM
    # model_name='SVM'
    # clf=svm.SVC(C=0.4,kernel='poly',random_state=7)
    # Bagging
    # model_name='Bagging_Classifier'
    # clf=BaggingClassifier(linear_model.LogisticRegression(C=1,penalty='l2',tol=1e-6),
    #                        n_estimators=16, max_samples=0.5,max_features=0.5,n_jobs=-1)
    # DT
    # model_name='decision_tree'
    # clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=1, min_samples_split=2,
    #                                   random_state=7)
    # NN
    model_name='MLP'
    clf = MLPClassifier(activation='relu', hidden_layer_sizes=(15, 2), random_state=7, solver='adam')
    clf.fit(x, y.astype('int'))

    # data washing test part
    data_test = pd.read_csv("test.csv")
    data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 2

    tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    null_age = tmp_df[data_test.Age.isnull()].values

    X = null_age[:, 1:]
    predictedAges = rfr.predict(X)
    data_test.loc[ (data_test.Age.isnull()), 'Age' ] = np.rint(predictedAges)

    data_test['Title'] = data_test['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
    set_age_type(data_test)
    set_SibSp_type(data_test)
    set_Parch_type(data_test)
    set_Cabin_type(data_test)
    set_family_size_type(data_test)
    set_Fare_bucket(data_test)
    set_title_type(data_test)
    set_MorC_type(data_test)

    dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
    dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')
    dummies_Age = pd.get_dummies(data_test['Age_bucket'], prefix='Age_bucket')
    dummies_SibSp = pd.get_dummies(data_test['SibSp_bucket'], prefix='SibSp')
    dummies_Parch = pd.get_dummies(data_test['Parch_bucket'], prefix='Parch')
    dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
    dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
    dummies_Family_size = pd.get_dummies(data_test['Family_size_category'], prefix='Family_size_category')
    dummies_Fare = pd.get_dummies(data_test['Fare_bucket'], prefix='Fare_bucket')
    dummies_Title= pd.get_dummies(data_test['Title_type'], prefix='Title')
    dummies_Cabin_class = pd.get_dummies(data_test['Cabin_class'], prefix='Cabin_class')

    test_df = pd.concat([data_test, dummies_Sex, dummies_Pclass,dummies_Age,dummies_SibSp,dummies_Parch,dummies_Embarked,
                         dummies_Cabin,dummies_Family_size,dummies_Fare,dummies_Title,dummies_Cabin_class], axis=1)
    test_df['Age_scaled'] = scaler.fit_transform(test_df[['Age']])
    test_df['Fare_scaled'] = scaler.fit_transform(test_df[['Fare']])
    test_df.drop(col[2:-2], axis=1, inplace=True)
    test_df.drop(['SibSp_0','Parch_0'], axis=1, inplace=True)
    # test
    test_df.to_csv("test_washed.csv", index=False)
    test_np=test_df.values
    predictions = clf.predict(test_np[:,1:])

    result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
    result.to_csv('result/'+model_name+"_predictions.csv", index=False)
    # print(pd.DataFrame({"columns": list(train_df.columns)[1:], "coef": list(clf.coef_.T)}))