# coding: utf-8
__version__: 1
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing

plt.rcParams['font.family'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    data_train = pd.read_csv('./train.csv')
    return data_train

def plot_contrib(data_train):

    fig = plt.figure()
    fig.set(alpha=0.2)

    plt.subplot2grid((2, 3), (0, 0))
    data_train.Survived.value_counts().plot(kind='bar')
    plt.title(u"获救情况")


    plt.ylabel(u'人数')
    plt.ylabel(u'人数')

    plt.subplot2grid((2, 3), (0, 1))
    data_train.Pclass.value_counts().plot(kind='bar')
    plt.ylabel(u'人数')
    plt.title(u"乘客等级分布")

    plt.subplot2grid((2, 3), (0, 2))
    plt.scatter(data_train.Survived, data_train.Age)
    plt.ylabel(u'年龄')
    plt.grid(b=True, which='major', axis='y')
    plt.title(u"按年龄看获救分布")

    plt.subplot2grid((2, 3), (1, 0), colspan=2)
    data_train.Age[data_train.Pclass == 1].plot(kind='kde')
    data_train.Age[data_train.Pclass == 2].plot(kind='kde')
    data_train.Age[data_train.Pclass == 3].plot(kind='kde')
    plt.xlabel(u'年龄')
    plt.ylabel(u'密度')
    plt.title(u'各等级的乘客年龄分布')
    plt.legend((u'一等舱', u'二等舱', u'三等舱'), loc='best')

    plt.subplot2grid((2, 3), (1, 2))
    data_train.Embarked.value_counts().plot(kind='bar')
    plt.title(u"各登船口岸上船人数")
    plt.ylabel(u'人数')
    plt.show()

def plot_Pclass_result_relation(data_train):
    # 绘制某些属性和最终结果的关联性
    fig = plt.figure()
    fig.set(alpha=0.2)

    Survived_0 = data_train.Pclass[data_train.Survived==0].value_counts()
    Survived_1 = data_train.Pclass[data_train.Survived==1].value_counts()
    df = pd.DataFrame({'获救': Survived_1, '未获救': Survived_0})
    print(df.head())
    df.plot(kind='bar', stacked=True)       # stacked=True  使每个等级的获救和未获救堆起来
    plt.title(u"各乘客等级的获救情况")
    plt.xlabel(u"乘客等级")
    plt.ylabel(u"人数")
    plt.show()

def plot_gender_result_relation(data_train):
    Survived_0 = data_train.Sex[data_train.Survived==0].value_counts()
    Survived_1 = data_train.Sex[data_train.Survived==1].value_counts()
    df = pd.DataFrame({'获救':Survived_1, '未获救':Survived_0})
    df.plot(kind='bar')
    plt.title('性别和获救情况')
    plt.show()

def plot_embarked_result_relation(data_train):
    Survived_0 = data_train.Embarked[data_train.Survived==0].value_counts()
    Survived_1 = data_train.Embarked[data_train.Survived==1].value_counts()
    df = pd.DataFrame({'获救':Survived_1, '未获救':Survived_0})
    df.plot(kind='bar')
    plt.title('登船港口和获救情况')
    plt.show()

def set_missing_ages(df):
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]        # 获取子数据集
    kown_age = age_df[age_df.Age.notnull()].values
    unkown_age = age_df[age_df.Age.isnull()].values
    # y是目标年龄
    y = kown_age[:, 0]
    X = kown_age[:, 1:]     # X 为属性值
    # 建模和训练
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    # 预测
    predict = rfr.predict(unkown_age[:, 1:])
    df.loc[(df.Age.isnull()), 'Age'] = predict      # 选取区域
    return df, rfr

def set_Cabin_type(df):
    df.loc[df.Cabin.notnull(), 'Cabin'] = "Yes"
    df.loc[df.Cabin.isnull(), "Cabin"] = "No"
    return df

def set_dummies(df):
    dummies_Cabin = pd.get_dummies(df['Cabin'], prefix='Cabin')
    dummies_Embarked = pd.get_dummies(df['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(df.Sex, prefix='Sex')
    dummies_Pclass = pd.get_dummies(df.Pclass, prefix='Pclass')
    df = pd.concat([df, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    return df

def data_standard(df):
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
    df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)
    fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)
    return df

def logisticRegression(df):
    from sklearn import linear_model
    train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.values
    y = train_np[:, 0]
    X = train_np[:, 1:]
    print(X.shape)
    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X, y)
    print(pd.DataFrame({'contr':list(train_df.columns)[1:], 'paramter': list(clf.coef_.T)}))
    return clf

def gain_train_data():
    df = load_data()
    df, rfr = set_missing_ages(df)
    df = set_Cabin_type(df)
    df = set_dummies(df)
    df = data_standard(df)
    return df, rfr

def test_data_preprocessing(rfr):
    df = pd.read_csv('./test.csv')
    df.loc[df.Fare.isnull(), 'Fare'] = 0
    # set missing ages
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]  # 获取子数据集
    unkown_age = age_df[age_df.Age.isnull()].values
    # 使用训练数据集上的随即森林模型预测
    predict = rfr.predict(unkown_age[:, 1:])
    df.loc[(df.Age.isnull()), 'Age'] = predict  # 选取区域
    df = set_Cabin_type(df)
    df = set_dummies(df)
    df = data_standard(df)
    # 获取需要的数据
    test_df = df.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    test_np = test_df.values
    return test_np, df.PassengerId.values

def main():
    train_data, rfr = gain_train_data()
    clf = logisticRegression(train_data)
    test_data, passengerId = test_data_preprocessing(rfr)
    predict = clf.predict(test_data)
    result = pd.DataFrame({'PassengerId':passengerId, 'Survived':predict.astype(np.int32)})
    result.to_csv('./logistic_regression_predictions.csv', index=False)




if __name__ == '__main__':
    main()