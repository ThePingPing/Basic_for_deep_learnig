import math
import random
from sympy.abc import y, t, z, x, c
from sympy import log
from matplotlib import pyplot as plt
import random as rd
import numpy as np
import pandas as pd

def serie_in_pandas():
    vec_to_ascii = []
    vec = np.arange(0, 10, 2)
    vec_number_ascii = np.arange(97, 102)
    for c0 in vec_number_ascii:
        vec_to_ascii.append(chr(c0))
    data = pd.Series(vec)
    data_value = data.values
    data_index = data.index
    data_slice = data[:3]
    data_new = pd.Series(data=vec, index=vec_to_ascii)
    print(data)
    print()
    print(data_value)
    print()
    print(data_index)
    print()
    print(data_slice)
    print(vec_to_ascii)
    print()
    print(data_new)


def diconory_in_pandas():
    list_math_scort = {'laura': 17, 'serge': 13.4, 'jacob': 15.2, 'lola': 12.5, 'eric': 20} # dictionaire
    panda_math = pd.Series(list_math_scort) # transforme to panda serie
    panda_math_slice = pd.Series(list_math_scort, ['jacob', 'lola']) # take a slice from a dic
    print(panda_math)
    print()
    print(panda_math_slice)


def modifie_serie_in_pandas():
    vec_to_ascii = []
    list_up_num = []
    list_up_index = []
    vec = np.arange(0, 10) # to give a vector of range 0-9
    vec_number_ascii = np.arange(97, 97+10) # range of ascii value
    for c0 in vec_number_ascii:
        vec_to_ascii.append(chr(c0)) # change a number to char
    data = pd.Series(vec, vec_to_ascii) # create a serie in pandas
    print(data)
    print()
    data_up = data.copy()
    for num in range(10, 20):
        list_up_num.append(num) ## to append in array value
    for letter in range(107, 117):
        list_up_index.append(chr(letter)) ## to append index in letter
    print(list_up_num)
    print(list_up_index)
    data_new_data = pd.Series(list_up_num, list_up_index) # create a new serie in panda
    print(data_new_data)
    data_final = pd.concat([data, data_new_data]) #concatenate tow serie dont forgot de []
    print(data_final)
    print(data_up)


def data_frame_in_pandas():
    list_math_scort = {'laura': 17, 'serge': 13.4, 'jacob': 15.2, 'lola': 12.5, 'eric': 20} # dictionaire
    list_history_scort = {'laura': 14, 'serge': 15, 'jacob': 16.7, 'lola': 20, 'eric': 9.5}
    list_physic_scort = {'laura': 13, 'serge': 16.4, 'jacob': 11.2, 'lola': 18.5, 'eric': 19}
    panda_math = pd.Series(list_math_scort)
    data_frame_math = pd.DataFrame(panda_math, columns=['Notes'])
    data_frame_all_score = pd.DataFrame([list_math_scort, list_history_scort, list_physic_scort])
    dics_frame_generate = [{'a': i, 'b': i*2, 'c': i*3} for i in range(10)] # generate a dictionary
    data_frame_dics = pd.DataFrame(dics_frame_generate) # take a dictionary and convert to Data frame
    print(data_frame_math)
    print()
    print(data_frame_all_score)
    print()
    print(dics_frame_generate)
    print()
    print(data_frame_dics)


def creat_data_frame_from_serie_pandas():
    dic_math_scort = {'laura': 17, 'serge': 13.4, 'jacob': 15.2, 'lola': 12.5, 'eric': 20}
    dic_physic_scort = {'laura': 13, 'serge': 16.4, 'jacob': 11.2, 'lola': 18.5, 'eric': 19}
    serie_frame_math = pd.Series(dic_math_scort)
    serie_frame_pysic = pd.Series(dic_physic_scort)
    data_frame_notes = pd.DataFrame({'Math': serie_frame_math, 'Physics': serie_frame_pysic}) # creat a frame from tow dics names and value organisate from the keys names
    data_frame_notes1 = pd.DataFrame([serie_frame_math, serie_frame_pysic])
    data_frame_notes_index = data_frame_notes.index # give the lign index from the data frame
    data_frame_notes_index_col = data_frame_notes.columns # give the col index from the data frame
    data_notes_from_col_math = data_frame_notes['Math']
    data_notes_from_col_phy = data_frame_notes['Physics']
    """print(serie_frame_math)
    print()
    print(serie_frame_pysic)
    print()
    print("FRom data frame notes ", '\n', data_frame_notes)
    print()
    print(data_frame_notes1)
    print()
    print(data_frame_notes_index)
    print()
    print(data_frame_notes_index_col)
    print()
    print(data_notes_from_col_math)
    print()
    print(data_notes_from_col_phy)"""
    return data_frame_notes



def data_frame_from_matrix_numpy():
    vec_random = np.random.randint(0, 100, 12)
    vec_arang = np.arange(10,22)
    matrix0 = np.resize(vec_arang, (3, 4))
    matrix = vec_random.reshape(3,4) # creat a random matrix from a random vector with size conresponding to the numbers of elemenets
    matrix1 = np.random.rand(3, 4) # create a matrix random in [0,1] interval and size (3,4)
    data_fram_from_matrix = pd.DataFrame(matrix1, columns=['col1', 'col2', 'clo3', 'clo4'], index=['lign1', 'lign2', 'lign3'])
    data_frame_matrix0 = pd.DataFrame(matrix0, columns=['col1', 'col2', 'clo3', 'clo4'], index=['lign1', 'lign2', 'lign3'])
    print(vec_random)
    print()
    print(matrix)
    print()
    print(matrix1)
    print()
    print(data_fram_from_matrix)
    print()
    print(matrix0)
    print(data_frame_matrix0)



def data_frame_change_data_in_panda(data_frame_arg):
    #print("FROM data_frame_change_data_in_panda", '\n', data_frame_arg)
    data_frame_arg_work = data_frame_arg # dont use copy coz it's another data frame automat
    data_frame_arg_work['Average'] = (data_frame_arg['Math']+data_frame_arg['Physics'])/2
    data_frame_arg_work_only_notes = data_frame_arg_work[['Math', 'Physics']]
    data_frame_drop_col = data_frame_arg_work.drop('Average', axis=1) # remove a cal coz the axis for the col is 1 and lign it's 0
    data_frame_drop_lign = data_frame_arg_work.drop('laura', axis=0) # remove a lign coz the axis for the col is 1 and lign it's 0

    """print(data_frame_arg_work, '\n')
    print(data_frame_arg_work_only_notes, '\n')
    print(data_frame_drop_col, '\n', data_frame_arg_work)
    #data_frame_arg.drop('Average', axis=1, inplace=True) # to revome the col Average in the original Data Frame
    print(data_frame_drop_lign, '\n')
    print(data_frame_arg_work)"""
    return data_frame_arg_work


def data_frame_selected_data_from_pandas(data_frame_arg):
    print("Hello From data_frame_selected_data_from_pandas ")
    print(data_frame_arg, '\n')
    data_frame_only_math = data_frame_arg.Math # can use '.label' to give the only one col you need also you can use data['label']
    data_frame_math_phy = data_frame_arg[['Math', 'Physics']] # not a choice to use [['Label1', 'Label2']] if you want a numbers of col
    data_frame_only_one_student_loc = data_frame_arg.loc['laura'] # use DataFrame.loc['Label'] to take only lign data
    data_frame_only_student_loc_lign = data_frame_arg.iloc[2] # use DataFrame.iloc[k] the index to take the lign in the data frame
    data_frame_only_student_and_works = data_frame_arg.loc[['laura', 'serge'], ['Math', 'Physics']] # if you want only somes works and somes students
    print(data_frame_only_math, '\n')
    print(data_frame_math_phy, '\n')
    print(data_frame_only_one_student_loc, '\n')
    print(data_frame_only_student_loc_lign, '\n')
    print(data_frame_only_student_and_works, '\n')


def data_frame_slice_in_panda():
    dic_math_scort = {'laura': 17, 'serge': 13.4, 'jacob': 15.2, 'lola': 12.5, 'eric': 20}
    dic_physic_scort = {'laura': 13, 'serge': 16.4, 'jacob': 11.2, 'lola': 18.5, 'eric': 19}
    dic_history_scort = {'laura': 12, 'serge': 13.8, 'jacob': 14.2, 'lola': 9.5, 'eric': 13}
    dic_gym_scort = {'laura': 18, 'serge': 13.7, 'jacob': 11.9, 'lola': 18.3, 'eric': 8.4}
    dic_latin_scort = {'laura': 14.1, 'serge': 17.3, 'jacob': 12.2, 'lola': 15.5, 'eric': 10}
    dic_informatic_scort = {'laura': 6.5, 'serge': 17.9, 'jacob': 15.7, 'lola': 18.9, 'eric': 15}

    data_frame_serie_math = pd.Series(dic_math_scort)
    data_frame_serie_physic = pd.Series(dic_physic_scort)
    data_frame_serie_history = pd.Series(dic_history_scort)
    data_frame_serie_gym = pd.Series(dic_gym_scort)
    data_frame_serie_latin = pd.Series(dic_latin_scort)
    data_frame_serie_informatics = pd.Series(dic_informatic_scort)


    data_frame = pd.DataFrame({'Math':data_frame_serie_math, 'Physics': data_frame_serie_physic, 'History': data_frame_serie_history,
                               'Gym': data_frame_serie_gym, 'Latin': data_frame_serie_latin, 'Info': data_frame_serie_informatics})
    data_frame_with_average = data_frame.copy()

    data_frame_with_average['Average'] = (data_frame['Math'] + data_frame['Physics'] + data_frame['History'] + data_frame['Gym'] + data_frame['Latin'] + data_frame['Info'])/(len(data_frame.columns))
    data_frame_cut_3_loc = data_frame_with_average.loc['laura':'jacob', :'Gym'] # take a cut from the student1 to the sudent i , and take the work from the start to the Label work
    data_frame_cut_3_iloc = data_frame_with_average.iloc[1:, :4] # can use also index iloc[lign = x:k)=non , col = y:z=non] more simple!!!
    """print(data_frame, '\n')
    print(data_frame_with_average, '\n')
    print(data_frame_cut_3_loc)
    print("this my len", len(data_frame.columns))
    print(data_frame_with_average, '\n')
    print(data_frame_cut_3_iloc)"""

    return data_frame_with_average


def data_frame_conditional_in_panda(data_frame_arg):
    print("from data frame data_frame_conditional_in_panda", '\n', data_frame_arg, '\n')
    condition_val = 17
    condition_val_phy = 13
    data_frame_bool = (data_frame_arg > condition_val) # True False DataFrame give From a data frame
    data_frame_bool_notes = data_frame_arg[data_frame_bool] # give the only original elemets respect the True Bool matrix use the original matrix for that
    data_frame_bool_phy = (data_frame_arg.iloc[ :, 1:2] > condition_val_phy)
   # data_frame_bool_notes_phy = data_frame_arg[[data_frame_arg['Physics'] > condition_val_phy]] # to give all Nan DataFrame with NAn and TRue value elements
    data_teste = data_frame_arg[data_frame_arg['Physics'] > condition_val_phy] # to take only a sub-DataFrame from the original data with respecte the condition
    data_frame_phy_cond_and_info_only = data_frame_arg[data_frame_arg['Physics'] > condition_val_phy] ['Info'] # take only the students respect also the condition and the Info works without the label
    data_frame_phy_cond_and_info_only_with_label = data_frame_arg[data_frame_arg['Physics'] > condition_val_phy] [['Info']]
    data_frame_phy_cond_and_info_only_with_label1 = data_frame_arg[data_frame_arg['Physics'] > condition_val_phy][
        ['Math', 'Physics', 'Info']] # take a numbers of col respect the conditions
    print(data_frame_bool, '\n')
    print(data_frame_bool_notes, '\n')
    print(data_frame_bool_phy, '\n')
    print(data_teste, '\n')
    print(data_frame_phy_cond_and_info_only,'\n')
    print(data_frame_phy_cond_and_info_only_with_label, '\n')
    print(data_frame_phy_cond_and_info_only_with_label1)



def data_frame_somes_conditional_in_panda(data_frame_arg_con):
    condition_val_math = 13
    condition_val_gym = 9
    serie_frame_bool_cond = (data_frame_arg_con['Math'] >= condition_val_math) & (data_frame_arg_con['Gym'] >= condition_val_gym)
    data_frame_bool = data_frame_arg_con[serie_frame_bool_cond] # give the Dataframe respect all condition

    print(data_frame_arg_con, '\n')
    print(serie_frame_bool_cond)
    print(data_frame_bool)


def data_frame_intinal_reparm_in_panda(data_frame_arg_con):
    students = []
    for i in range(5):
        students.append('students'+str(i))
    print("hello",students)
    data_frame_reindex = data_frame_arg_con

    data_frame_reindex['Students'] = (students)
    data_frame_reindex_new = data_frame_reindex.set_index('Students')

    data_frame_std0 = data_frame_reindex_new.loc['students0']
    print(data_frame_arg_con, '\n')
    print(data_frame_reindex, '\n')
    print(data_frame_reindex_new, '\n')
    print(data_frame_std0)


def multi_index_in_pandas():
    full_big_ab = []
    full_samal_a = []
    full_data = []
    constant_big_a = 65
    constant_big_b = 66
    for i in range(3):
        full_big_ab.append(chr(constant_big_a)) # creat the variable for the Dataframe Good idea to make like that
        full_samal_a.append('a'+str(i))
        full_data.append('data'+str(i+1))
    for i in range(2):
        full_big_ab.append(chr(constant_big_b))
        full_samal_a.append('b'+str(i))
    df1 = pd.DataFrame(np.random.rand(5, 3), index=[full_big_ab, full_samal_a], columns=full_data) # creat a data frame with random variable btween [0,1] and attribute the label of ligne and col

    """"print("hello From multi_index_in_pandas ")
    print(full_big_ab, '\n', full_samal_a, '\n', full_data)
    print(df1)"""
    return df1


def take_ele_from_mul_index_df_in_pandas(df_mult_ind_arg):

    df_only_A = df_mult_ind_arg.loc['A']
    df_only_A_a1 = df_mult_ind_arg.loc['A'].loc['a1'] # take in A level and a1 lign
    df_only_A_a1_simple =  df_mult_ind_arg.loc['A', 'a1']
    df_only_A_a1_col_index = df_mult_ind_arg.loc['A', 'a1']['data3'] # take only the value in the level A lign a1 and col Data3
    #df_mult_ind_arg.index.names = ['Nivo1', 'Nivo2']

    print(df_mult_ind_arg, '\n')
    print(df_only_A, '\n')
    print(df_only_A_a1, '\n')
    print(df_only_A_a1_simple,'\n')
    print(df_only_A_a1_col_index)
    #print(df_mult_ind_arg)



def combinaison_data_frame_panda():

    dic_paris_info = {'laura': 17, 'serge': 13.4, 'jacob': 15.2, 'lola': 12.5, 'eric': 20}
    dic_marseille_info = {'laura': 13, 'serge': 16.4, 'jacob': 11.2, 'lola': 18.5, 'eric': 19}
    dic_lyon_info = {'laura': 12, 'serge': 13.8, 'jacob': 14.2, 'lola': 9.5, 'eric': 13}

    serie_paris = pd.Series(dic_paris_info)
    serie_marseille = pd.Series(dic_marseille_info)
    serie_lyon = pd.Series(dic_lyon_info)

    data_frame_info = pd.DataFrame({'Paris': serie_paris, 'Marseille': serie_marseille, 'Lyon': serie_lyon}) # rappele quand tu fais un Data frame apartire d une serie
    ########################################################################################################## les nom donner dans le data frame son les colone atribue au tableaux final

    #print(serie_paris, '\n' , dic_lyon_info)
    #print('\n', data_frame_info)


    data_info1 = {'Paris': [75000, 285000, 321, 154, 276, 422] ,'Mareseille': [92000, 199330, 221, 157, 376, 522],
                 'Lyon': [34000, 232800, 222, 254, 322, 362]}
    data_info2 = {'Avinion': [43000, 85000, 321, 23000], 'Bordeaux': [65000, 732800, 222, 27800]}

    df_info1 = pd.DataFrame(data_info1, index=['code postal', 'populations', 'universiter', 'ecole secondaire', 'ecole primaire', 'jardin enfants'])
    df_info2 = pd.DataFrame(data_info2, index=['code postal', 'populations', 'universiter', 'nombres eleves'])

    df_info_union = pd.concat([df_info1, df_info2], axis=1)## by default the concat is per ligne so i have to change the axis to 1 so it's per col

    df_info_union_comun = pd.concat([df_info1, df_info2], axis=1, join='inner')

    print(df_info1,'\n')
    print(df_info2, '\n')
    print()
    print()
    print(df_info_union, '\n')
    print(df_info_union_comun)


def groupe_in_panda():
    full_big_abc = []
    value_numbers = []
    constant_big_a = 65
    constant_big_b = 66
    constant_big_c = 67

    for i in range(2):
        full_big_abc.append(chr(constant_big_a))
        full_big_abc.append(chr(constant_big_b))
        full_big_abc.append(chr(constant_big_c)) # creat the variable for the Dataframe Good idea to make like that
    for i in range(6):
        value_numbers.append(i)
    df1 = pd.DataFrame({'Key': full_big_abc, 'data': value_numbers})

    group_df1_key = df1.groupby('Key') # to regroupe the sqme elemente have the same Key
    group_df1_key_sum = group_df1_key.sum() # to some all elements in the same key value
    group_df1_key_averag = group_df1_key.mean()

    print(df1, '\n')
    print(group_df1_key_sum, '\n')
    print(group_df1_key_averag)

    ##################################################################################################################

    vec_names = ['Amandine', 'Serge', 'Clara', 'Lola', 'Celine', 'Dan']
    vec_math = ['Math']
    vec_phy = ['Physique']
    vec_math_phy = []
    vec_names_double = []
    vec_math_note = []
    vec_phy_note = []
    vec_all_notes = []
    print()

    print(2*len(vec_names))


    for i in range(2):
        for c0 in vec_names:
            vec_names_double.append(c0)

    for i in range(6):
        vec_math_note.append(round(random.uniform(9, 20), 1))
        vec_phy_note.append(round(random.uniform(9, 20), 1))
        for c0 in vec_math:
            vec_math_phy.append(c0)

    for i in range(6):
        for c1 in vec_phy:
            vec_math_phy.append(c1)

    for c0 in vec_math_note:
        vec_all_notes.append(c0)
    for c0 in vec_phy_note:
        vec_all_notes.append(c0)

    print(vec_math_phy,'\n')
    print(vec_math_note, "\n")
    print(vec_math_phy, '\n')
    print(vec_phy_note, '\n')
    print(vec_all_notes, '\n')

    df_to_all = pd.DataFrame({'Noms': vec_names_double, 'Matieres': vec_math_phy, 'Notes': vec_all_notes})

    df_to_all_name = df_to_all.groupby('Noms')
    df_to_all_name_sum = df_to_all_name.sum()
    df_to_all_name_average = df_to_all_name.mean()
    df_to_all_name_std = df_to_all_name.std()
    df_to_all_name_all_fonction = df_to_all_name.describe() # give count (number of occurence for the student), the average , std...
    df_only_dan = df_to_all_name_all_fonction.loc['Dan']

    print(df_to_all, '\n')

    dan_loc = df_to_all.iloc[5]
    print(dan_loc)
    print(df_to_all_name, '\n')
    print(df_to_all_name_sum, '\n')
    print(df_to_all_name_average, '\n')
    print(df_to_all_name_std, '\n')
    print(df_to_all_name_all_fonction)
    print(df_only_dan)


def cross_dynamic_table_panda():

    ascii_M = 77
    ascii_F = 70
    vec_age = []
    vec_genders = []
    vec_traitement_jours = []
    vec_prix = []
    vec_bin = []
    vec_localisation = ['R', 'U', 'SU', 'U', 'U', 'SU', 'R', 'U', 'R', 'R']

    for i in range(10):

        vec_age.append(random.randint(9, 99))
        vec_traitement_jours.append(random.randint(1, 31))
        vec_bin.append(random.randint(0,1))
        vec_prix.append(random.randint(1500, 6700))

        if (i % 7 != 0 | i % 3 != 0):
            vec_genders.append(chr(ascii_F))
        else:
            vec_genders.append(chr(ascii_M))

    print(vec_age, '\n', vec_genders, '\n', vec_bin, '\n', vec_prix, '\n', vec_traitement_jours, '\n')

    df_meta = pd.DataFrame({'age': vec_age, 'gendres': vec_genders, 'durer de traitement': vec_traitement_jours, 'prix': vec_prix,
                            'guerrison': vec_bin, 'localisation': vec_localisation})

    df_meta_mf_mean = df_meta.groupby('gendres')[['age', 'durer de traitement']].mean()


    print(df_meta,'\n')
    print(df_meta_mf_mean, '\n')
    df_table_pivot = pd.pivot_table(df_meta, index=['gendres'])# take lign index name , and only col numerical
    df_table_pivot_guerison = df_meta.pivot_table('guerrison', index='gendres', columns='localisation')
    print(df_table_pivot,'\n')
    print(df_table_pivot_guerison)


def operation_in_panda(data_frame_arg):
    def carre(x):
        power_x = x ** 2
        return power_x

    print(data_frame_arg)
    data_frame = data_frame_arg

    data_frame_umath = data_frame['Math'].unique() # give only numerical data from the col specifie
    data_frame_umath_power = data_frame['Math'].apply(carre)
    data_frame_umath_sort = data_frame.sort_values('Math')# to sort the data frame in intersting ('name col)
    data_frame_null_bool = data_frame.isnull()
    data_frame_set_id_name = data_frame['Math'].idxmax() # give the col and returne the personne with the max notes
    data_frame_set_arg = data_frame['Math'].argmax() # give the col and returne the personne with the max notes
    print(data_frame_umath,'\n')
    print(data_frame_umath_power,'\n')
    print(data_frame_umath_sort, '\n')
    print(data_frame_null_bool, '\n')
    print(data_frame_set_id_name, '\n')
    print(data_frame_set_arg)




def data_analysis_panda():
    x=0


if __name__ == "__main__":

    #serie_in_pandas()
    #diconory_in_pandas()
    #modifie_serie_in_pandas()
    #data_frame_in_pandas()
    #creat_data_frame_from_serie_pandas()
    #data_frame_from_matrix_numpy()
    #data_frame_arg = creat_data_frame_from_serie_pandas()
    #data_frame_change_data_in_panda(data_frame_arg)
    #data_frame_selected_arg = data_frame_change_data_in_panda(data_frame_arg)
    #data_frame_selected_data_from_pandas(data_frame_selected_arg)
    data_frame_arg_con = data_frame_slice_in_panda()
    #data_frame_conditional_in_panda(data_frame_arg_con)
    #data_frame_somes_conditional_in_panda(data_frame_arg_con)
    #data_frame_intinal_reparm_in_panda(data_frame_arg_con)
    #df_mult_ind_arg = multi_index_in_pandas()
    #take_ele_from_mul_index_df_in_pandas(df_mult_ind_arg)
    #combinaison_data_frame_panda()
    #groupe_in_panda()
    #cross_dynamic_table_panda()
    #operation_in_panda(data_frame_arg_con)
    data_analysis_panda()



