#include "mfWarp.h"

int php_mf_my_train(char * tr_path,char * model_path)
{
	int result = mf::mf_my_train(tr_path, model_path);
	//printf("here %s", tr_path);
	return result;
	/*printf("train file: %s", tr_path);
	return 1;*/
}

float * php_utility_train(float * train_data,int  train_triplet_num,double p_l2 ,double q_l2 ,int k ,int iters , double eta,int *lens)
{
        float * result = mf::utility_train(train_data,train_triplet_num,p_l2 ,q_l2,k,iters,eta, *lens);
        return result;
}

float * php_utility_predict(float * test_arr,int  test_triplet_num,float  *model_arr,int model_arr_len)
{
        float * result = mf::utility_predict(test_arr, test_triplet_num,  model_arr,model_arr_len);
        return result;
}

float * php_cos_similarity(int item_id, float * q_arr, int q_arr_num)
{
        float * cos_res = mf::cos_similarity(item_id,q_arr,q_arr_num);
        return cos_res;
}

int * php_DINA(float * q_arr,int q_triplet_num, float * x_arr,int x_triplet_num,int iterators)
{
       int * result = mf::DINA(q_arr, q_triplet_num, x_arr,x_triplet_num,iterators);
       return result;
}
