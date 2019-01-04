#ifndef _LIBMF_H
#define _LIBMF_H

#include <utility>

#if (defined WIN32 || defined _WIN32)
#if defined(MF_EXPORTS) || defined(mf_EXPORTS)
#define MF_API __declspec(dllexport)
#else
#define MF_API __declspec(dllimport)
#endif
#elif defined __GNUC__ && __GNUC__ >= 4
#define MF_API __attribute__ ((visibility ("default")))
#else
#define MF_API
#endif

#ifdef __cplusplus
//extern "C"
//{

namespace mf
{
#endif

typedef float mf_float;
typedef double mf_double;
typedef int mf_int;
typedef long long mf_long;

enum {P_L2_MFR=0, P_L1_MFR=1, P_KL_MFR=2, P_LR_MFC=5, P_L2_MFC=6, P_L1_MFC=7,
      P_ROW_BPR_MFOC=10, P_COL_BPR_MFOC=11};
enum {RMSE=0, MAE=1, GKL=2, LOGLOSS=5, ACC=6, ROW_MPR=10, COL_MPR=11,
      ROW_AUC=12, COL_AUC=13};

struct mf_node
{
    mf_int u;
    mf_int v;
    mf_float r;
};

struct mf_problem
{
    mf_int m;
    mf_int n;
    mf_long nnz;
    struct mf_node *R;
};

struct mf_parameter
{
    mf_int fun;
    mf_int k;
    mf_int nr_threads;
    mf_int nr_bins;
    mf_int nr_iters;
    mf_float lambda_p1;
    mf_float lambda_p2;
    mf_float lambda_q1;
    mf_float lambda_q2;
    mf_float eta;
    bool do_nmf;
    bool quiet;
    bool copy_data;
};

MF_API struct mf_parameter mf_get_default_param();

struct mf_model
{
    mf_int fun;
    mf_int m;
    mf_int n;
    mf_int k;
    mf_float b;
    mf_float *P;
    mf_float *Q;
};

MF_API mf_problem read_problem(char const * path);

MF_API mf_int mf_save_model(struct mf_model const *model, char const *path);

MF_API struct mf_model* mf_load_model(char const *path);

MF_API void mf_destroy_model(struct mf_model **model);

MF_API struct mf_model* mf_train(
    struct mf_problem const *prob,
    struct mf_parameter param);

MF_API mf_int mf_my_train(char const * tr_path, char const * model_path);

MF_API float * utility_train(float * train_data,
	                int  train_triplet_num,
	                double p_l2 ,
	                double q_l2 ,
	                int k ,
	                int iters ,
	                double eta,
					        int &lens);

MF_API float * utility_predict(float * test_arr,
	                      int  test_triplet_num,
	                      float  *model_arr,
                        int model_arr_len);

MF_API float *  cos_similarity(int item_id, float * q_arr, int q_arr_num);

MF_API int *  DINA(float * q_arr,int q_triplet_num, float * x_arr,int x_triplet_num,int iterators);

MF_API struct mf_model* mf_train_on_disk(
    char const *tr_path,
    struct mf_parameter param);

MF_API struct mf_model* mf_train_with_validation(
    struct mf_problem const *tr,
    struct mf_problem const *va,
    struct mf_parameter param);

MF_API struct mf_model* mf_train_with_validation_on_disk(
    char const *tr_path,
    char const *va_path,
    struct mf_parameter param);

MF_API mf_double mf_cross_validation(
    struct mf_problem const *prob,
    mf_int nr_folds,
    struct mf_parameter param);

MF_API mf_double mf_cross_validation_on_disk(
    char const *prob,
    mf_int nr_folds,
    mf_parameter param);

MF_API mf_float mf_predict(struct mf_model const *model, mf_int u, mf_int v);

MF_API mf_double calc_rmse(mf_problem *prob, mf_model *model);

MF_API mf_double calc_mae(mf_problem *prob, mf_model *model);

MF_API mf_double calc_gkl(mf_problem *prob, mf_model *model);

MF_API mf_double calc_logloss(mf_problem *prob, mf_model *model);

MF_API mf_double calc_accuracy(mf_problem *prob, mf_model *model);

MF_API mf_double calc_mpr(mf_problem *prob, mf_model *model, bool transpose);

MF_API mf_double calc_auc(mf_problem *prob, mf_model *model, bool transpose);

#ifdef __cplusplus
} // namespace mf

//} // extern "C"
#endif

#endif // _LIBMF_H
