import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from xgboost import XGBClassifier
import optuna
import string
from sklearn import preprocessing
import category_encoders as ce


def readData():
    df_out = pd.read_csv("train.csv",header=0)
    return df_out

def preprocess(df):
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Embarked'] = df['Embarked'].fillna('Unknown')
    df['Embarked'] = df['Embarked'].map({'S':0, 'C':1, 'Q':2, 'Unknown':3}).astype(int)
    df['Sex'] = df['Sex'].apply(lambda x: 1 if x =='male' else 0)
    df['Cabin'] = df['Cabin'].fillna(0)
    df['Cabin'] = df['Cabin'].apply(lambda x: 1 if x !=0 else 0)
    df['Name'] = convertName(df['Name'])
    df = df.drop(['PassengerId','Ticket'], axis = 1)
    return df

def convertName(pd):
    con = []
    for el in pd:
        if el.find('Mr.') >= 0:
            con.append(1)
        elif el.find('Miss.') >= 0:
            con.append(2)
        elif el.find('Mrs.') >= 0:
            con.append(3)
        else:
            con.append(0)
    return con


def convertTicket(pd):
    t_data = pd['Ticket'].apply(lambda x: x.strip('0123456789'))
    t_data = t_data.fillna('None')
    ce_oe = ce.OrdinalEncoder(cols=['Ticket'],handle_unknown='impute')
    print(dir(ce_oe))
    #df_session_ce_ordinal = ce_oe.fit_transform(df_session)
    #df_session_ce_ordinal.head()


d = readData()
convertTicket(d)