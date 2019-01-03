import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from xgboost import XGBClassifier
import optuna
import string
from sklearn import preprocessing
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder

'''
	PassengerId	Survived	Pclass	Name	Sex	Age	SibSp	Parch	Ticket	Fare	Cabin	Embarked
0	1	0	3	Braund, Mr. Owen Harris	male	22.0	1	0	A/5 21171	7.2500	NaN	S
1	2	1	1	Cumings, Mrs. John Bradley (Florence Briggs Th...	female	38.0	1	0	PC 17599	71.2833	C85	C
2	3	1	3	Heikkinen, Miss. Laina	female	26.0	0	0	STON/O2. 3101282	7.9250	NaN	S
3	4	1	1	Futrelle, Mrs. Jacques Heath (Lily May Peel)	female	35.0	1	0	113803	53.1000	C123	S
4	5	0	3	Allen, Mr. William Henry	male	35.0	0	0	373450	8.0500	NaN	S
'''



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
    ce_oe = LabelEncoder()
    con = ce_oe.fit_transform(t_data)
    return con


d = readData()
print(convertTicket(d))