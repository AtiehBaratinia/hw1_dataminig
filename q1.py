import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
import seaborn as sb

def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x == 0 else x for x in values]


def histogram_plot(covid):

    x = list(covid.groupby('confirmed_date', sort=False).groups.keys())

    y = covid.groupby('confirmed_date', sort=False).size()

    plt.xticks(rotation='vertical')
    plt.title('number of patients per day')
    plt.bar(x, y)
    plt.show()


def scatter_plot():
    x = covid.groupby('birth_year').groups.keys()

    p = covid.pivot_table(index='state',
                          columns='birth_year',
                          values='id',
                          fill_value=0,
                          aggfunc='count').unstack()  # .to_frame().rename(columns={0: 'count'})

    p_array = p.to_numpy()
    isolated_people = np.array([])
    deceased_people = np.array([])
    released_people = np.array([])
    i = 0
    while i < len(p_array):
        deceased_people = zero_to_nan(np.append(deceased_people, [p_array[i]]))
        isolated_people = zero_to_nan(np.append(isolated_people, [p_array[i + 1]]))
        released_people = zero_to_nan(np.append(released_people, [p_array[i + 2]]))
        i += 3
    ys = [deceased_people, isolated_people, released_people]
    colors = itertools.cycle(["r", "yellow", "g"])
    for y in ys:
        plt.scatter(x, y, color=next(colors))
    plt.xlabel('birth_year')
    plt.ylabel('count')
    plt.legend(["deceased_people", "isolated_people", "released_people"])

    plt.show()


def matrix_plot(covid):
    birth = list(covid.groupby('birth_year').groups.keys())
    p = covid.pivot_table(index='sex',
                          columns='birth_year',
                          values='id',
                          fill_value=0,
                          aggfunc='count').unstack().to_frame().rename(columns={0: 'count'})

    p_array = p.to_numpy()
    male_people_year = np.array([])
    female_people_year = np.array([])

    i = 0
    while i < len(p_array):
        male_people_year = list(np.append(male_people_year, [p_array[i]]))
        female_people_year = list(np.append(female_people_year, [p_array[i + 1]]))
        i += 2
    data = {'birth_year': birth, 'birth_year_of_male': male_people_year, 'birth_year_of_female': female_people_year}

    df = pd.DataFrame(data, columns=['birth_year', 'birth_year_of_male', 'birth_year_of_female'])

    # selecting three numerical features
    features = ['birth_year','birth_year_of_male', 'birth_year_of_female']

    # plotting the scatter matrix
    # with the features
    scatter_matrix(df[features])
    plt.show()


if __name__ == "__main__":
    # to show all the data
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    covid = pd.read_csv('data/covid.csv')

    # show the table of data
    print(covid)
    # show the names of columns
    print(covid.columns)
    # show the number of data
    print('the number of rows is ', covid.shape[0])
    # show the std, mean, avg of birth_year
    print(covid.birth_year.describe())

    # drop any rows that have NaN(means null)
    print(covid.info())
    covid_without_null = covid.dropna(how='any')
    print(covid_without_null)
    print(covid_without_null.shape[0])
    sb.pairplot(covid)

    # plots
    histogram_plot(covid)
    matrix_plot(covid)
    scatter_plot()
