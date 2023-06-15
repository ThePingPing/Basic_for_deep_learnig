import math
from sympy.abc import y, t, z, x, c
from sympy import log
from matplotlib import pyplot as plt
import random as rd
import numpy as np



def creat_vec(vec):
    dim = vec.ndim
    dim1 = np.ndim(vec)
    line_colone = np.shape(vec)
    vec_size = np.size(vec)
    vec_10 = np.arange(10)
    print(vec_10)
    #vec_10.resize((3, 4), refcheck=False) // Modifie surplace
    print("my dimention is:" + str(dim) + str(dim1) + str(line_colone) + str(vec_size))
    for c0 in vec:
        print(c0)
    vec_reshap = vec_10.reshape((2, 5))
    vec_to_resize = np.resize(vec_10, (5, 5))
    print(vec_reshap)
    print()
    print(vec_to_resize)

def unions_vecteur(vec1,vec2 , vec3):
    print("form union Concatenate")
    vec_c1 = np.concatenate([vec1, vec2])
    vec_c2 = np.concatenate([vec2, vec3])
    matrix1 = vec_c1.reshape((2, 3))
    matrix2 = vec_c2.reshape((2, 3))
    matrix1_c = np.concatenate((matrix1, matrix2), axis=0)
    matrix2_c = np.concatenate((matrix1, matrix2), axis=1)
    print(vec_c1)
    print()
    print(vec_c2)
    print()
    print(matrix1_c)
    print()
    print(matrix2_c)


def creat_matrix_np():

    print("from Create Matrix np")

    matrix_zero = np.zeros((5, 5))
    matrix_full = np.full((4, 4), 5, int)
    matrix_id = np.eye(4)
    print(matrix_zero)
    print()
    matrix_one = np.ones((5, 5), int)
    print(matrix_one)
    print()
    print(matrix_full)
    print()
    print(matrix_id)


def take_diagonal():

    matrix_diag = np.arange(1, 17).reshape((4, 4))
    diag1 = np.diag(matrix_diag, k=0)
    diag2= np.diag(matrix_diag, k=1)
    diag3 = np.diag(matrix_diag, k=-1)
    mat_diag1 = np.diag(diag1)
    print(diag1)
    print()
    print(diag2)
    print()
    print(diag3)
    print()
    print(mat_diag1)

def takelinspace():
    matrix_lin = np.linspace(0, 5, 10) # star in zero done in 5 and equidistance between every elements
    print(matrix_lin)
    print()


def random_set():
    set_rand = np.random.rand(10) # number of element
    print(set_rand)
    print()
    mat_rand = np.random.rand(5, 5) # stipulate the size
    print(mat_rand)
    print()
    mat_gauss = np.random.randn(5) # number of element gauss distribution
    print(mat_gauss)
    print()
    x = np.random.randint(0,100)
    print(x)
    vec_random = np.random.randint(0, 100, 10)
    print(vec_random)
    print()
    mat_random = np.random.randint(0, 100, (5, 5))
    print(mat_random)
    print()
    mat_nor_sigma2 = np.random.normal(loc=10, scale=2, size=(3, 4))
    print(mat_nor_sigma2)
    print()
    counter = 0
    for c0 in mat_nor_sigma2[:]:
        counter += c0
    print(counter)


def just_test():
    x = 0
    set_test = np.arange(0, 10)
    print(set_test)
    for c0 in set_test:
        x += c0
    print(x)


def calculate_in_numpy():
    vec_random = np.random.randint(0, 100, 15)
    mat_random = np.random.randint(0, 100, 15).reshape((3, 5))
    max_vec_col = mat_random.max(axis=0) # take the vector max from all col elements
    max_vec_lign = mat_random.max(axis=1) # take the vector max from all lign elements
    max_in_mat = mat_random.max() # take the max element in the matrix
    min_in_mat = mat_random.min() # take the min element in the matrix
    max_value = vec_random.max() # take the max element in the vector (vec_ramdom = vector)
    min_value = vec_random.min() # take the min element in the vector
    index_max_value = vec_random.argmax() # take the index for thr max elements in the vector
    average_mat_col = mat_random.mean(axis=0) # take average from the col in the matrix
    average_mat_lign = mat_random.mean(axis=1) # take average from the lign in the matrix
    average_mat_all = mat_random.mean() # take average from the all elements in the matrix
    sum_mat_col = mat_random.sum(axis=0) # take sum from the col in the matrix
    sum_mat_lign = mat_random.sum(axis=1) # take sum from the lign in the matrix
    sum_mal_all = mat_random.sum() # take sum from the all elements in the matrix
    std_mat_col = mat_random.std(axis=0) # take sigma(std) from the col in the matrix
    std_mat_lign = mat_random.std(axis=1) # take sigma(std) from the lign in the matrix
    std_mat_all = mat_random.std() # take sigma(std) from the all elements in the matrix
    print()
    print(max_value)
    print()
    print(min_value)
    print()
    print(index_max_value)
    print()
    print(mat_random)
    print()
    print(max_vec_col)
    print()
    print(max_vec_lign)
    print()
    print(max_in_mat)
    print()
    print(min_in_mat)
    print()
    print(average_mat_col)
    print()
    print(average_mat_lign)
    print()
    print(average_mat_all)
    print()
    print(sum_mat_col)
    print()
    print(sum_mat_lign)
    print()
    print(sum_mal_all)
    print()
    print(std_mat_col)
    print()
    print(std_mat_lign)
    print()
    print(std_mat_all)


def slicing_in_numpy():
    constant = 77
    vec_range_chang_val = []

    vec_to_start = np.random.randint(0, 100, 30)
    vec_slice = vec_to_start[:5] # take the the elements to the strat include and 5 index non included
    vec_range_chang_val_copy = vec_to_start.copy() # explicite copy for anohter array
    vec_range_chang_val_copy[:5] = constant
    for c0 in vec_to_start:
        vec_range_chang_val.append(c0)

    matrix_start = np.arange(5, 50, 5).reshape((3, 3))
    print(matrix_start)
    mat_ling_0 = matrix_start[0, :]
    mat_ling_1 = matrix_start[1, :]
    mat_ling_2 = matrix_start[len(matrix_start)-1, :]
    mat_col_0 = matrix_start[:, 0]
    mat_col_1 = matrix_start[:, 1]
    mat_col_2 = matrix_start[:, len(matrix_start)-1]
    mat_cut_lign_col = matrix_start[0, 1:] # take the elemente ine the first lign and only from the second col to the end




    print(vec_to_start)
    print()
    print(vec_slice)
    print()
    print(vec_to_start)
    print(vec_range_chang_val)
    print()
    print(vec_range_chang_val_copy)
    print()
    print(matrix_start)
    print()
    print(mat_ling_0)
    print()
    print(mat_ling_1)
    print()
    print(mat_ling_2)
    print()
    print(mat_col_0)
    print()
    print(mat_col_1)
    print()
    print(mat_col_2)
    print()
    print(mat_cut_lign_col)


def condition_in_numpy():
    condition_num = 5
    condition_num_mat = 14
    vec = np.arange(0, 15)
    matrix0 = np.arange(0, 30).reshape((5, 6))
    vec_check_cond = vec > condition_num # to have a array with True or False from the elements respecte de condition
    vec_condition_element = vec[vec_check_cond] # to have a array numbers only respect the condition
    matrix0_check_bool = matrix0 > condition_num_mat #to have a matrix with True or False from the elements respecte de condition
    matrix0_check_bool_mod2 = (matrix0 > condition_num_mat) & (matrix0 % 2 == 0)
    vec_elem_matrix0_true = matrix0[matrix0_check_bool] # to have a array numbers only respect the condition
    vec_elem_matrix0_true_mod2 = matrix0[matrix0_check_bool & (matrix0 % 2 == 0)] # to combine 2 condition and take all elements
    print(vec)
    print()
    print(vec_check_cond)
    print()
    print(vec_condition_element)
    print()
    print(matrix0_check_bool)
    print()
    print(vec_elem_matrix0_true)
    print()
    print(matrix0_check_bool_mod2)
    print()
    print(vec_elem_matrix0_true_mod2)


def change_condition_numpy():
    change_num = 77
    matrix0 = np.arange(0, 30).reshape((5, 6))
    vec_even = matrix0[matrix0 % 2 == 0] # vector for all element even
    matrix0_even_change_num = matrix0.copy() # copy a matrix
    matrix0_even_change_num[matrix0_even_change_num % 2 == 0] = change_num # copy to the start matrix and change all even to change num

    print(matrix0)
    print()
    print(vec_even)
    print()
    print(matrix0_even_change_num)
    print()
    print(matrix0)


def binary_change_numpy():
    flag0 = 0.5
    matrix_random = np.resize([rd.random() for i in range(25)], (5, 5))
    matrix_random_binary = matrix_random.copy()
    matrix_random_binary[matrix_random_binary > flag0] = 1
    matrix_random_binary[matrix_random_binary <= flag0] = 0
    print(matrix_random)
    print(matrix_random_binary)


def algebra_in_numpy():
    constante = 5
    v_vec = np.arange(0, 10)
    u_vec = np.arange(10, 20)
    vu_sum_vec = v_vec + u_vec # sum tow vector
    v_vec_mult_const = v_vec * constante # mult a vect by a constante
    v_vec_power = v_vec ** 2 # to give the vecteur in power
    v_vec_exp = np.exp(v_vec) # give all element in vector in e^ v[i]
    print(v_vec)
    print()
    print(u_vec)
    print()
    print(vu_sum_vec)
    print()
    print(v_vec_mult_const)
    print()
    print(v_vec_power)
    print()
    print(v_vec_exp)


def matrix_op_numpy():
    matrix_op1 = np.resize(np.arange(0, 10), (3, 3)) # resize use cut if you are out of range number in the matrix
    matrix_op2 = np.resize(np.arange(11, 20), (3, 3))
    matrix_sum = matrix_op1 + matrix_op2 # sum btw 2 matrix
    matrix_mult = np.dot(matrix_op1 , matrix_op2)
    matrix_op1_trans = np.transpose(matrix_op1)
    matrix_op2_invers = np.linalg.inv(matrix_op2)
    matrix_uniti = np.dot(matrix_op2 , matrix_op2_invers)
    matrix_op1_eig_vev_val = np.linalg.eig(matrix_op1) # give a double tuple with eight value and vectors
    print(matrix_op1)
    print()
    print(matrix_op2)
    print()
    print(matrix_sum)
    print()
    print(matrix_mult)
    print()
    print(matrix_op1_trans)
    print()
    print(matrix_op1)
    print(matrix_op2_invers)
    print()
    print(matrix_uniti)
    print()
    print(matrix_op1_eig_vev_val)



if __name__ == "__main__":
    vec1 = [1, 2, 3]
    vec2 = [4, 5, 6]
    vec3 = [7, 8, 9]
    #vec_all = [vec1, vec2, vec3]
    #matrix = np.array(vec_all)
    #creat_vec(matrix)
    #unions_vecteur(vec1, vec2, vec3)
    #creat_matrix_np()
    #take_diagonal()
    #takelinspace()
    #random_set()
    #just_test()
    #calculate_in_numpy()
    #slicing_in_numpy()
    #condition_in_numpy()
    #change_condition_numpy()
    #binary_change_numpy()
    #algebra_in_numpy()
    matrix_op_numpy()

