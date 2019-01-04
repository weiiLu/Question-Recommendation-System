#pragma once
#include <iostream>
#include <string>
#include "../mf/mf.h"

extern "C" int php_mf_my_train(char * tr_path,char * model_path);
extern "C" float * php_utility_train(float * train_data,int  train_triplet_num,double p_l2 ,double q_l2 ,int k ,int iters ,double eta ,int *lens );
extern "C" float * php_utility_predict(float * test_arr,int  test_triplet_num,float  *model_arr,int model_arr_len);
extern "C" float * php_cos_similarity(int item_id, float * q_arr, int q_arr_num);
extern "C" int * php_DINA(float * q_arr,int q_triplet_num, float * x_arr,int x_triplet_num,int iterators);
