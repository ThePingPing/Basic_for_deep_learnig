import math
import random
import seaborn as sns
import sympy as sy
from sympy.abc import y, t, z, x, c
from sympy import log
from matplotlib import pyplot as plt
import random as rd
import numpy as np
import pandas as pd


def open_file_csv_seaborn():
    students = pd.read_csv('student.csv')
    col_val = students.columns.values
    plt.figure(figsize=(10, 8))
    #sns.displot(students.Note1, bins=50, kde=True)
    #sns.stripplot(x='Note1', data=students)
    sns.pointplot(x='sex', y='Note1', data=students)
    plt.show()
    print(col_val)
    """print(students.head())
    print(col_val)"""


def fonction_capplot_seaborn():

    students = pd.read_csv('student.csv')
    plt.figure(figsize=(10, 8))
    tsns.catplot(x='Note1', data=students, kind='boxen') # creat a pseudo mustache plot
    sns.catplot(x='Note1', data=students, kind='srip') # create a plot with connection
    sns.catplot(x='Note1', data=students, kind='point') # create a lign with the medianne
    sns.catplot(x='Note1', data=students, kind='count') # create a sticked diagrame
    plt.show()


def fonction_panda_seaborn():

    students = pd.read_csv('student.csv')
    table_frequence = pd.crosstab(students.sex, 'frequence', normalize=True)
    table_pourcentage = table_frequence.assign(frequence=(100*table_frequence.frequence))
    table_pourcentage_graph = table_frequence.assign(sex=table_pourcentage.index, frequence=(100*table_frequence.frequence))
    plt.figure(figsize=(5, 4))
    sns.barplot(x='sex', y='frequence', data=table_pourcentage_graph)
    plt.show()
    print(table_frequence, '\n')
    print(table_pourcentage)


def linear_reg_seaborn():

    students = pd.read_csv('student.csv')
    print(students.head())
    col_val = students.columns.values
    print(col_val)
    plt.figure(figsize=(5, 4))
    sns.jointplot(x='absences', y='Note3', data=students, kind='reg') # to cross tow section in the database and under ply give a plot
    sns.pairplot(data=students, vars=['Medu', 'Fedu', 'studytime', 'absences', 'Note3'], kind='reg') # give all linear regretion 2 with 2 under plot that's give all plot
    sns.pairplot(data=students, vars=['Medu', 'Note2', 'studytime', 'absences', 'Note3'], hue='sex', kind='reg')
    sns.catplot(x='sex', y='Note3', data=students, kind='bar')# cross between numerical and quality
    sns.catplot(x='Fjob', y='Note3', hue='sex', data=students, kind='bar') #the bar in the graph it's the hue parm
    sns.catplot(x='Fjob', y='Note3', hue='sex', data=students, kind='violin', split=True)

    plt.show()


def iris_cnv_seaborn():
    iris = pd.read_csv('iris.csv')
    col_iris = iris.columns.values
    correlation_iris = iris.corr()
    plt.figure(figsize=(5,4))
    sns.heatmap(correlation_iris, annot=True, cmap='seismic', linewidths=4, linecolor='w') # exemple to create a heat map
    sns.lmplot(x='sepal.length', y='petal.length', data=iris, hue='variety', markers=['o', 'p', 'v'])# in the same window a plot correlation frome the varitey betwen spal and petal
    sns.lmplot(x='sepal.length', y='petal.length', data=iris, col='variety') #plot quant numerbes of col='name'
    plt.show()
    print(iris.head(), '\n')
    print(col_iris, '\n')
    print(iris['variety'].unique())


def dynamic_table_seaborn():

    vol_data = pd.read_csv('vols.csv')
    vol_data_head = vol_data.head()
    vol_data_col = vol_data.columns.values
    vol_year = vol_data['year'].unique()
    corrs_dynamic_table = vol_data.pivot_table(index='month', columns='year', values='passengers')
    print(vol_data, '\n', vol_data_head, '\n', vol_data_col, "\n")
    print(vol_year,'\n')
    print(corrs_dynamic_table)


def lol():
    iris = pd.read_csv('iris.csv')
    print(iris)

    plt.figure(figsize=(5, 4))
    sns.displot(iris['sepal.length'])

    plt.show()


if __name__ == "__main__":

    #open_file_csv_seaborn()
    #fonction_capplot_seaborn()
    #fonction_panda_seaborn()
    linear_reg_seaborn()
    #iris_cnv_seaborn()
    #dynamic_table_seaborn()
    #lol()
