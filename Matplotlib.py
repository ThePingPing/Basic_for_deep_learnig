import math
import random
import sympy as sy
from sympy.abc import y, t, z, x, c
from sympy import log
import matplotlib as mpl
import matplotlib.pyplot as plt
import random as rd
import numpy as np
import pandas as pd
from sympy.physics.control.control_plots import matplotlib


def use_representation_mlp():
    vec_time = [i for i in np.arange(1, 11)]
    vec_concentration = []


    for i in range(len(vec_time)):
        vec_concentration.append(round(random.uniform(5.5, 20), 1))

    graph_arg = plt.scatter(vec_time, vec_concentration, marker='o')
    #plt.show()

    #print("hello")
    #print(vec_time)
    #print(vec_concentration,'\n')

    tranf =[]
    tranf.append(vec_time)
    tranf.append(vec_concentration)
    print(tranf)
    return tranf


def use_fonction_mpl(arg_val):

    vec_time = arg_val[0]
    vec_concentration = arg_val[1]
    print(vec_time,'\n', vec_concentration)
    ggraph = plt.plot(vec_time, vec_concentration, marker='o', color='r')
    plt.show()


def use_axes_titel_mpl():
    data_val = np.linspace(0, 20, 100)
    print(data_val)
    #plt.plot(data_val, data_val, linestyle='-')
    plt.title('representation graph')
    plt.xlabel('absis axe')
    plt.ylabel('value axe')
    y = np.linspace(-20, 20, 100)
    plt.plot(y, np.cos(y), linestyle="-", color='b', label='cos')# build a graph for the fonction with x= absis from linespace
    plt.plot(y, np.sin(y), linestyle="-", color='r', label='sin')
    #plt.savefig('coscin1.pdf', dpi=200)# save the graph in pdf
    plt.legend()# use label in the fonction argument to show a legend on the plot

    plt.xlim(-5,15)# border the absis
    plt.ylim(-2,2)# border f(x)
    plt.axis([-5, 5, -2, 2])
    plt.axis('off')# remove absis x and y if i want to back just write 'one'
    plt.show()  # to show the plot


def axes_control_limit_mpl():
    fig = plt.figure()
    fig.set_facecolor('red')
    axes0 = plt.axes()
    axes1 = plt.axes([0.5, 0.6, 0.25, 0.25]) # (0.5) = start axes in 50% from the original axe (0.6) = 60% from the higth lageure = 25%
    plt.show()


def separet_plot_mpl():

    y = np.linspace(-20, 20, 100)
    """plt.plot(y, np.cos(y), linestyle="-", color='b',label='cos')  # build a graph for the fonction with x= absis from linespace
    plt.plot(y, np.sin(y), linestyle="-", color='r', label='sin')

    fig = plt.figure()
    first_plt_axe = fig.add_axes([0.1, 0.5, 0.8, 0.4]) # ca fonction pas tres bien
    second_plt_axe = fig.add_axes([0.1, 0, 0.8, 0.4]) # ca ne devise pas tres bien le fenetre
    first_plt_axe.plot(y, np.cos(y))
    second_plt_axe.plot(y, np.sin(y), color='r')

    fig_size_control = plt.figure(figsize=(10, 8))# control the sizes of all subplot from eatch subplot(hauteur largeur)
    fig_size_control.subplots_adjust(hspace=0.5, wspace=0.4)# control de l espacement vertical et lateral

    for i in range(1, 10):# to creat somes plot
        plt.subplot(3, 3, i) # nombres of lign , nombers of clom , nombers of elemets
        plt.text(0.5, 0.5, i, fontsize=16) # put the index of the plot in the midelle"""

    fig1, axe = plt.subplots(3, 2, sharex=True, sharey=True)
    plt.show()

    """for i in range(0, 2):
        for j in range(i):
            axe0[i, j].plot(x**i)"""


def historgramme_plot_mpl():

    age = []
    vec_bins_space = [i for i in np.arange(10, 105, 5)]
    print(vec_bins_space)
    for i in range(100):
        age.append(random.randint(4, 100))
    print(age)

    fig = plt.figure(figsize=(12, 8))
    plt.hist(age, bins=vec_bins_space, color='m', histtype='barstacked', edgecolor="black", alpha=0.5, rwidth=0.4)
    plt.show()


def historgramme_compar_plot_mpl():

    vec_ran1_normal = []
    vec_ran2_normal = []
    vec_ran3_normal = []
    vec_bins = [i for i in np.arange(0, 11, 1/5)]

    x = np.random.normal(0, 0.8, 1000)
    y = np.random.normal(-3, 1, 1000)
    z = np.random.normal(4, 2, 1000)

    vec_ran1_normal = vec_ran1_normal + [x]
    vec_ran2_normal = vec_ran2_normal + [y]
    vec_ran3_normal = vec_ran3_normal + [z]

    style = dict(histtype='stepfilled', alpha=0.5, bins=vec_bins)

    fig = plt.figure(figsize=(10, 8))
    plt.hist(vec_ran1_normal, label="distribustion1", **style)
    plt.hist(vec_ran2_normal, label="distribustion2", **style)
    plt.hist(vec_ran3_normal, label="distribustion3", **style)
    plt.legend()
    plt.show()
    print(vec_ran1_normal, '\n', vec_ran2_normal, '\n', vec_ran3_normal)


def cloud_3d_plot_mpl():
    size_point = 40
    vec_cloud_x = []
    vec_cloud_y = []
    vec_cloud_z = []
    for i in range(100):
        vec_cloud_x.append(random.normalvariate(0, 1))
        vec_cloud_y.append(random.normalvariate(0, 1))
        vec_cloud_z.append(random.normalvariate(0, 1))

    """print(vec_cloud_x,'\n')
    print(vec_cloud_y,'\n')
    print(vec_cloud_z,'\n')"""


    """plt.figure(figsize=(10, 8))
    axes = plt.axes(projection='3d')

    axes.scatter3D(vec_cloud_x, vec_cloud_y, vec_cloud_z, s=size_point)
    axes.set_xlabel('axe x')
    axes.set_ylabel('axe y')
    axes.set_zlabel('axe z')"""

    x_val = np.linspace(0, 20, 200)
    y_val = np.cos(2*x_val)
    z_val = np.sin(2*x_val)
    print(x_val)

    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    ax.plot3D(x_val, y_val, z_val)
    plt.show()


def fonction_3D_mpl():

    def sin_cos(x, y):
        return np.cos(x) + np.sin(y)

    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)

    X, Y = np.meshgrid(x, y)

    Z = sin_cos(X, Y)

    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='plasma')
    ax.set_xlabel('axe x')
    ax.set_ylabel('axe y')
    ax.set_zlabel('axe z')
    plt.show()








if __name__ == "__main__":

    #arg_val = use_representation_mlp()
    #use_fonction_mpl(arg_val)
    #use_axes_titel_mpl()
    #axes_control_limit_mpl()
    #separet_plot_mpl()
    #historgramme_plot_mpl()
    #historgramme_compar_plot_mpl()
    #cloud_3d_plot_mpl()
    fonction_3D_mpl()