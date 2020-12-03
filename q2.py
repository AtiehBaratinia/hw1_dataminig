import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import random
import numpy as np
from pandas.plotting import scatter_matrix
import sklearn
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    # to show all the data
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    students = pd.read_csv('data/student.csv', delimiter=';')



    test = pd.DataFrame(columns=students[:].columns)
    train = pd.DataFrame.copy(students)

    print(students.info())

    # 80 are for testing, the rest are for training
    for i in range(79):
        f = random.randint(0, 310)
        test = test.append(train.iloc[f], ignore_index=True)
        train = train.drop(train.index[f])
        train = train.reset_index(drop=True)

    # show the relationship between G1, G2, G3
    # scatter_matrix(students[['G1', 'G2', 'G3']])
    # plt.show()

    students_data = train.iloc[:, [30,31]].values
    students_target = train.iloc[:, 32].values

    students_data_name = ['G1', 'G2']
    X, y = students_data, students_target
    LinReg = LinearRegression().fit(X, y)
    print(LinReg.score(X, y))
    print(LinReg.coef_)

    predict = np.array([[]])
    i = 0
    while i < 79:
        predict = np.append(predict, [LinReg.predict(np.array([[test['G1'][i], test['G2'][i]]]))])
        i += 1

    s = np.mean((predict - test['G1']) ** 2)
    print('Mean squared error: ', s)
