#include "../mf/mf.h"
#include <iostream>
int main(int argc, char *argv[])
{
	//mf::mf_my_train("D:/Funny-Works/VSCommonTest/libmf-2.01/demo/real_matrix.tr.txt",
	//			"D:/Funny-Works/VSCommonTest/libmf-2.01/demo/real_matrix.mod.txt");
  float train_data[24] = {
						0, 0, 5 ,
						0, 2, 10,
						0, 3, 2	,
						1, 0, 7	,
						1, 1, 3	,
						1, 3, 0	,
						2, 1, 2	,
						2, 3, 9
					};
	float test_arr[18] = {
						0 ,0,
						0 ,2,
						0 ,3,
						1 ,0,
						1 ,1,
						1 ,3,
						2 ,1,
						2 ,3,
						2 ,2 };

	float q_arr[75] = {0,0,1,
		0 ,1, - 1,
		0 ,2 ,- 1,
		0 ,3, - 1,
		0 ,4, - 1,
		1, 0, - 1,
		1 ,1,   1,
		1 ,2 ,- 1,
		1 ,3 ,1,
		1 ,4 ,- 1,
		2 ,0, - 1,
		2 ,1 ,- 1,
		2 ,2, - 1,
		2 ,3 ,- 1,
		2 ,4 ,1,
		3 ,0 ,1,
		3 ,1 ,- 1,
		3 ,2 ,1,
		3 ,3 ,1,
		3 ,4 ,- 1,
		4 ,0 ,1,
		4 ,1 ,- 1,
		4 ,2 ,1,
		4 ,3 ,- 1,
		4 ,4, - 1 };
	float x_arr[60] = {
		0 ,0 ,1,
		0 ,1 ,0,
		0 ,2 ,1,
		0 ,3 ,0,
		0 ,4 ,1,
		1 ,0 ,1,
		1 ,1 ,0,
		1 ,2 ,1,
		1 ,3 ,1,
		1 ,4 ,0,
		2 ,0 ,1,
		2 ,1 ,0,
		2 ,2 ,0,
		2 ,3 ,0,
		2 ,4 ,1,
		3 ,0 ,0,
		3 ,1 ,0,
		3 ,2 ,1,
		3 ,3 ,0,
		3 ,4 ,1};
  int model_lens;
  float *model_array;
  model_array = mf::utility_train(train_data, 8, 0.1, 0.1, 8, 30, 0.1, model_lens);
	float * predict = mf::utility_predict(test_arr,9,model_array,model_lens);
	int *dina_res = mf::DINA(q_arr,25 ,x_arr,20,2);
	//output
  std::cout << "train cuccessfully!" << std::endl;
	std::cout<< "model array:" << std::endl;
	for (int i = 0; i < model_lens; i++)
	{
		std::cout<<float(model_array[i]) << std::endl;
	}

	std::cout<< "predict value:" << std::endl;
	for (int i = 0; i < 9; i++)
	{
		std::cout<<float(predict[i]) << std::endl;
	}

  std::cout<< "dina_res value:" << std::endl;
	for (int i = 0; i < 20; i++)
	{
		std::cout<< dina_res[i] ;
		if((i+1)%5==0)
		{
			  std::cout<< std::endl;
		}
	}

	return 0;
}
