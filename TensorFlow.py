import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import sklearn
import tensorflow as tf
from sklearn import datasets
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.utils import to_categorical
from sklearn.datasets import make_blobs
from datetime import datetime as dt
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def basic_tensor_flow():
    print("Hello Tensorflow")
    tensor_zero_dim = tf.constant(4)
    print(tensor_zero_dim)

    ## arrays
    tensor_one_dim = tf.constant([2, 0, -3])
    print(tensor_one_dim)

    tensor_tow_dim = tf.constant([[1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9],
                                 [10, 11, 12]])
    print(tensor_tow_dim)

    tensor_three_dim = tf.constant([[[1, 2, 3], [-1, -2, -3]],
                                   [[4, 5, 6], [-4, -5, -6]],
                                   [[7, 8, 9], [-7, -8, -9]],
                                   [[10, 11, 12], [-10, -11, -13]]])

    print(tensor_three_dim)

    ### change the type inpute
    tensor_one_dim_float = tf.constant([2, 0, -3], dtype=tf.float32)
    print(tensor_one_dim_float)
    ## or you can casting a tensor alredy exist

    tensor_tow_dim = tf.cast(tensor_tow_dim, dtype=tf.float32)
    print(tensor_tow_dim)

    tensor_boolean = tf.constant([True, False, True, True, True, False])

    print(tensor_boolean)

    ## tensor String

    tensor_string = tf.constant(["Hello Word ", "NextWord", "NeXXt"])
    print(tensor_string)


def use_numpy_and_tensor():
    np_array = np.array([1, 2, 4])
    covert_tensor = tf.convert_to_tensor(np_array)
    print(covert_tensor)

    tensor_eye = tf.eye(

        num_rows = 3,
        num_columns = None,
        batch_shape = None,
        dtype = tf.float32,
        name = None
    )

    print(tensor_eye)

    tensor_eye_next = tf.eye(num_rows=5, num_columns=None, batch_shape=[2, ], dtype=tf.float32, name=None)

    print(tensor_eye_next)

    tensor_fill = tf.fill(dims=[3, 4], value=5, name=None)
    print(tensor_fill)

    ### all One Tensor

    tensor_one = tf.ones(shape=[5, 3])
    print(tensor_one)
########################################################################################################################
    tensor_three_dim = tf.constant([[[1, 2, 3], [-1, -2, -3]],
                                    [[4, 5, 6], [-4, -5, -6]],
                                    [[7, 8, 9], [-7, -8, -9]],
                                    [[10, 11, 12], [-10, -11, -13]]])


    tensor_like = tf.ones_like(tensor_three_dim)
    print(tensor_like)


    #### All zero Tensor
    tensor_zero = tf.zeros(shape=[5, 3])
    print(tensor_zero)

def random_tensor():

    tensor_random = tf.random.normal(shape = [3, 2], mean = 0.0, stddev = 1.0,
                                    dtype = tf.float32, seed = None, name = None)
    print(tensor_random)
    """You can change the mean and the Std , to keep random value close to the mean input """

    tensor_random_index = tf.random.normal(shape=[10, ], mean=20.0, stddev=5.0,
                                        dtype=tf.float32, seed=None, name=None)
    print(tensor_random_index) ## take All
    print(tensor_random_index[0:4]) ## 4 first element
    print(tensor_random_index[1:]) ## bounding
    print(tensor_random_index[1:6:2]) ## skipe evry tow element

    ## slice tow domention

    tensor_tow_dim = tf.constant([[1, 2, 3],
                                  [4, 5, 6],
                                  [7, 8, 9],
                                  [10, 11, 12]])

    print(tensor_tow_dim)

    print(tensor_tow_dim[0:3, 0:2])


def function_tensor():
############# Operation Beetwen Some Tensor#######################
    X_abs = tf.constant([-2.25, 3.25])

    tensor_tow_dim = tf.constant([[1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9],
                                 [10, 11, 12]])
    tf.abs(X_abs)

    x1 = tf.random.normal(shape=[10, ], mean=20.0, stddev=5.0, dtype=tf.float32, seed=None, name=None)
    x2 = tf.random.normal(shape=[10, ], mean=20.0, stddev=5.0, dtype=tf.float32, seed=None, name=None)
    X_const = tf.constant([7], dtype=tf.float32)
    tensor_add = tf.add(x1, x2)
    tensor_sub = tf.subtract(x1, x2)
    tensor_divid = tf.divide(x1, x2)
    tensor_multi = tf.multiply(x1, X_const)
    print(x1, x2)
    print(tensor_add)
    print(tensor_sub)
    print(tensor_divid)

    print(tensor_multi)

    X_const_multi = tf.constant([[7], [4], [5]], dtype=tf.float32)

    tensor_multi_line = tf.multiply(x1, X_const_multi)
    print(tensor_multi_line)

    ############# Min Max ArgMax Tensor#######################

    tensor_data = tf.random.normal(shape=[3, 6], mean=20.0, stddev=5.0, dtype=tf.float32, seed=None, name=None)
    X_argmax = tf.argmax(tensor_data)
    X_argmin = tf.argmin(tensor_data)
    print(tensor_data, "the ArgMax indexes:", X_argmax)
    print(tensor_data, "the ArgMin indexes:", X_argmin)

    base_tensor = tf.constant([[2, 2], [3, 3]])
    exp_tensor = tf.constant([[3, 0], [1, 4]])

    result_pow = tf.pow(base_tensor, exp_tensor)


    print(result_pow)

    sum_element_tensor = tf.reduce_sum(input_tensor=tensor_tow_dim, axis=None, keepdims=False, name=None) ## change the axis to 0 if you want sum the colum ,
                                                                                                        # and 1 to sum the row , with none he sum all element
    print(sum_element_tensor)

    tensor_top_k = tf.math.top_k(tensor_tow_dim) ## return the max value on each line on the Tensor
    print(tensor_top_k)

def linear_algebra_tensor():
    x1_mat = tf.random.normal(shape=[3, 6], mean=20.0, stddev=5.0, dtype=tf.float32, seed=None, name=None)
    x2_mat = tf.random.normal(shape=[6, 3], mean=20.0, stddev=5.0, dtype=tf.float32, seed=None, name=None)


    print(x1_mat, x2_mat)

    result_mult_mat = x1_mat @ x2_mat
    print(result_mult_mat)

    x1_mat_transpo = tf.transpose(x1_mat)
    print(x1_mat_transpo)

    A = np.random.randint(10, size=(2, 3, 4))
    B = np.random.randint(10, size=(2, 4, 5))
    print(A, '\n')
    print(B, '\n')

    C = np.matmul(A, B)
    print(C)


    print()


def common_function_tensor():

    tensor_three_dim = tf.convert_to_tensor(np.random.randint(10, size=(4, 2, 3)), dtype=tf.float32)
    #print(tensor_three_dim)

    #print(tf.expand_dims(tensor_three_dim, axis=0))

    ####### Reshape Tensor #######

    x_to_reshape = tf.constant([[3, 5, 6, 6], [4, 6, -1, 2]])
    #print(x_to_reshape)
    x_reshape = tf.reshape(x_to_reshape, [8]) ## it's not inplace And the reshape have to be conrespond for evry combination of number in the initial element
    #print(x_reshape)

    t1 = tf.convert_to_tensor(np.random.randint(10, size=(2, 3)), dtype=tf.float32)
    t2 = tf.convert_to_tensor(np.random.randint(10, size=(2, 3)), dtype=tf.float32)
    #print(t1, '\n', t2, '\n')


    t_concat_col = tf.concat([t1, t2], axis=0)
    t_concat_row = tf.concat([t1, t2], axis=1)


    #print(t_concat_col, '\n', t_concat_row)

    t_stack = tf.stack([t1, t2], axis=0) ## that's to create a new Axis he put the tensor under the first tensor and the new Axes depend the number for tensort you give
    #print(t_stack)

    t3 = tf.convert_to_tensor(np.random.randint(10, size=(2, 3)), dtype=tf.int32)
    t_padding = tf.constant([[1, 1], [2, 2]])

    #print(t3, '\n', t_padding)

    t_pad_result = tf.pad(tensor=t3, paddings=t_padding,  mode="CONSTANT", constant_values=-2) ## he take the padding and he put the tensort in the centre ,
    # he padding row up and dow in the first array you give, and col left and col right from the second array you give , that have to be in int32
    #print(t_pad_result)

    t4 = tf.convert_to_tensor(np.random.randint(10, size=(3, 1, 3)), dtype=tf.int32)
    print(t4.shape)
    print(t4)
    t5 = tf.constant([[2, 2, 2], [3, 3, 3], [4, 4, 4]])
    print(t5.shape)
    t_raged_tensor = tf.RaggedTensor.from_tensor(t5, lengths=[1, 0, 3])
    print(t4, '\n',t_raged_tensor )


def sparse_tensor():

    tensor_to_sparse = tf.sparse.SparseTensor(indices=[[1, 1], [3, 4]], values=[11, 56], dense_shape=[5, 6]) ## by default he put 0 value everywhere the index are not define
    tensor_sparse = tf.sparse.to_dense(tensor_to_sparse)

    print(tensor_sparse)

    #### to Change specificly value on a tensor###


def string_tensor():
    tensor_string = tf.constant(["Hello", "im ", "a ", "String"])
    print(tensor_string)

    tensor_sentence = tf.strings.join(tensor_string, separator=" ")
    print(tensor_sentence)


if __name__ == '__main__':
    #basic_tensor_flow()
    #use_numpy_and_tensor()
    #random_tensor()
    #function_tensor()
    #linear_algebra_tensor()
    #common_function_tensor()
    #sparse_tensor()
    string_tensor()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
