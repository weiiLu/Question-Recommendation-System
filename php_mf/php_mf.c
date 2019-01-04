/*
  +----------------------------------------------------------------------+
  | PHP Version 7                                                        |
  +----------------------------------------------------------------------+
  | Copyright (c) 1997-2017 The PHP Group                                |
  +----------------------------------------------------------------------+
  | This source file is subject to version 3.01 of the PHP license,      |
  | that is bundled with this package in the file LICENSE, and is        |
  | available through the world-wide-web at the following url:           |
  | http://www.php.net/license/3_01.txt                                  |
  | If you did not receive a copy of the PHP license and are unable to   |
  | obtain it through the world-wide-web, please send a note to          |
  | license@php.net so we can mail you a copy immediately.               |
  +----------------------------------------------------------------------+
  | Author:                                                              |
  +----------------------------------------------------------------------+
*/

/* $Id$ */



#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "php_ini.h"
#include "ext/standard/info.h"
#include "php_mf.h"


#ifdef PHP_WIN32
#endif

extern int php_mf_my_train(char * tr_path,char * model_path);
extern float * php_cos_similarity(int item_id, float * q_arr, int q_arr_num);
extern float * php_utility_train(float * train_data,int  train_triplet_num,double p_l2 ,double q_l2 ,int k ,int iters ,double eta ,int *lens);
extern float * php_utility_predict(float * test_arr,int  test_triplet_num,float  *model_arr,int model_arr_len);
extern int * php_DINA(float * q_arr,int q_triplet_num, float * x_arr,int x_triplet_num,int iterators);
/* If you declare any globals in php_php_libmf.h uncomment this:
ZEND_DECLARE_MODULE_GLOBALS(php_libmf)
*/

/* True global resources - no need for thread safety here */
static int le_php_mf;

/*
PHP_FUNCTION(generate_triplet_array)
{
	zval *input_table, *element;
	zend_bool bfill = 0;
	double fill_value = -1;

	if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "a|bd", &input_table, &bfill, &fill_value) == FAILURE) {
		return;
	}
  //RETURN_STRING("Hello Bug");
	int table_count, element_count;
	HashTable *table_hash, *element_hash;
	HashPosition pointer, element_point;

	int error_flag = 0;
	if (Z_TYPE_P(input_table) == IS_ARRAY)
	{
		error_flag = -1;

		table_hash = Z_ARRVAL_P(input_table);
		table_count = zend_hash_num_elements(table_hash);

		zval column1, column2;
		array_init(&column1);
		array_init(&column2);

		int valid = 1;

    //RETURN_STRING("Hello Bug");

		//Do some check work and get the column1 and column2
		char *label_name[3] = { NULL, NULL, NULL };
		for (zend_hash_internal_pointer_reset_ex(table_hash, &pointer);
			 zend_hash_has_more_elements_ex(table_hash, &pointer) == SUCCESS;
			 zend_hash_move_forward_ex(table_hash, &pointer)) {
			element = zend_hash_get_current_data_ex(table_hash, &pointer);

			switch (Z_TYPE_P(element))
			{
			case IS_ARRAY:
				element_hash = Z_ARRVAL_P(element);
				if (zend_hash_num_elements(element_hash) == 3)
				{

          //RETURN_STRING("Hello Bug");

					zval *sub_element, sub_key;
					int label_idx = 0;
					for (zend_hash_internal_pointer_reset_ex(element_hash, &element_point);
						 zend_hash_has_more_elements_ex(element_hash, &element_point) == SUCCESS;
						 zend_hash_move_forward_ex(element_hash, &element_point), label_idx++) {
						zend_hash_get_current_key_zval_ex(element_hash, &sub_key, &element_point);

						if (Z_TYPE(sub_key) == IS_STRING)
						{
							char *key_str = (*(Z_STR(sub_key))).val;
							if (key_str)
							{

                //RETURN_STRING("Hello Bug");

								sub_element = zend_hash_get_current_data_ex(element_hash, &element_point);
								//php_printf("%d </br>", Z_LVAL_P(sub_element));
								if (Z_TYPE_P(sub_element) != IS_LONG && Z_TYPE_P(sub_element) != IS_DOUBLE)
								{
									valid = 0;
									error_flag = 5;
									break;
								}

								switch (label_idx)
								{
								case 0:
									add_index_long(&column1, Z_LVAL_P(sub_element), 0);
									break;
								case 1:
									add_index_long(&column2, Z_LVAL_P(sub_element), 0);
									break;
								default:
									break;
								}
							}

							if (label_name[label_idx] && strcmp(label_name[label_idx], key_str) != 0)
							{
								valid = 0;
								error_flag = 4;
								break;
							}
							else
								label_name[label_idx] = key_str;
						}
						else
						{
							valid = 0;
							error_flag = 3;
							break;
						}
					}
				}
				else
				{
					valid = 0;
					error_flag = 2;
				}
				break;
			//case IS_OBJECT:
		 //break;
			default:
				valid = 0;
				error_flag = 1;
				break;
			}

			if (!valid)
				break;
		}

    //RETURN_STRING("Hello Bug");

		//Inverse column1 and column2 and set the true value of column1 and column2
		HashTable *column_hash;
		HashPosition column_pointer;
		zval column1_inv, column2_inv, key;
		int rows, cols;
		array_init(&column1_inv);
		array_init(&column2_inv);
		int inv_idx = 0;
		column_hash = Z_ARRVAL_P(&column1);

    //RETURN_STRING("Hello Bug");

		for (zend_hash_internal_pointer_reset_ex(column_hash, &column_pointer);
			 zend_hash_has_more_elements_ex(column_hash, &column_pointer) == SUCCESS;
			 zend_hash_move_forward_ex(column_hash, &column_pointer), inv_idx++)
		{
      //RETURN_STRING("Hello Bug");
			element = zend_hash_get_current_data_ex(column_hash, &column_pointer);

      //RETURN_STRING("Hello Bug");

			zend_hash_get_current_key_zval_ex(column_hash, &key, &column_pointer);

      //xxxxxxxxxx

			add_index_long(&column1_inv, inv_idx, Z_LVAL_P(&key));
			add_index_long(&column1, Z_LVAL_P(&key), inv_idx);
		}
		rows = inv_idx;

    //xxxxxxxxxx

		inv_idx = 0;
		column_hash = Z_ARRVAL_P(&column2);
		for (zend_hash_internal_pointer_reset_ex(column_hash, &column_pointer);
			 zend_hash_has_more_elements_ex(column_hash, &column_pointer) == SUCCESS;
			 zend_hash_move_forward_ex(column_hash, &column_pointer), inv_idx++)
		{
			element = zend_hash_get_current_data_ex(column_hash, &column_pointer);
			zend_hash_get_current_key_zval_ex(column_hash, &key, &column_pointer);

			add_index_long(&column2_inv, inv_idx, Z_LVAL_P(&key));
			add_index_long(&column2, Z_LVAL_P(&key), inv_idx);
		}
		cols = inv_idx;

    //xxxxxxxxx

		//Get triplets array , the error check is not needed here
		HashTable *column1_hash, *column2_hash;
		column1_hash = Z_ARRVAL_P(&column1);
		column2_hash = Z_ARRVAL_P(&column2);
		zval triplet_array;
		array_init(&triplet_array);

    // xxxxxxxxx
		if (bfill)
		{

			int matrix_count = rows * cols;
			double *densematrix = (double *)malloc(sizeof(double) * matrix_count);
			int m_idx;
			for (m_idx = 0; m_idx < matrix_count; m_idx++)
				densematrix[m_idx] = fill_value;

			for (zend_hash_internal_pointer_reset_ex(table_hash, &pointer);
				 zend_hash_has_more_elements_ex(table_hash, &pointer) == SUCCESS;
				 zend_hash_move_forward_ex(table_hash, &pointer)) {

				element = zend_hash_get_current_data_ex(table_hash, &pointer);
				element_hash = Z_ARRVAL_P(element);

				zval *sub_element, sub_key;
				int label_idx = 0;
				int row_idx, col_idx;
				double value;
				for (zend_hash_internal_pointer_reset_ex(element_hash, &element_point);
					 zend_hash_has_more_elements_ex(element_hash, &element_point) == SUCCESS;
					 zend_hash_move_forward_ex(element_hash, &element_point), label_idx++) {

					sub_element = zend_hash_get_current_data_ex(element_hash, &element_point);

					zend_long id;
					zval *find_result;
					switch (label_idx)
					{
					case 0:
						id = Z_LVAL_P(sub_element);
						//php_printf("%d ", Z_LVAL_P(sub_element));
						find_result = zend_hash_index_find(column1_hash, id);
						row_idx = Z_LVAL_P(find_result);
						break;
					case 1:
						id = Z_LVAL_P(sub_element);
						//php_printf("%d ", Z_LVAL_P(sub_element));
						find_result = zend_hash_index_find(column2_hash, id);
						col_idx = Z_LVAL_P(find_result);
						break;
					case 2:
						value = Z_TYPE_P(sub_element) == IS_LONG ? Z_LVAL_P(sub_element) : Z_DVAL_P(sub_element);
						break;
					default:
						break;
					}
				}

				int pos = row_idx * cols + col_idx;
				densematrix[pos] = value;
			}

			int r, c;
			for (r = 0, m_idx = 0; r < rows; r++)
			{
				for (c = 0; c < cols; c++, m_idx++)
				{
					add_next_index_long(&triplet_array, r);
					add_next_index_long(&triplet_array, c);
					add_next_index_double(&triplet_array, densematrix[m_idx]);
				}
			}
		}
		else
		{
			for (zend_hash_internal_pointer_reset_ex(table_hash, &pointer);
				 zend_hash_has_more_elements_ex(table_hash, &pointer) == SUCCESS;
				 zend_hash_move_forward_ex(table_hash, &pointer)) {

				element = zend_hash_get_current_data_ex(table_hash, &pointer);
				element_hash = Z_ARRVAL_P(element);

				zval *sub_element, sub_key;
				int label_idx = 0;
				int row_idx, col_idx;
				double value;
				for (zend_hash_internal_pointer_reset_ex(element_hash, &element_point);
					 zend_hash_has_more_elements_ex(element_hash, &element_point) == SUCCESS;
					 zend_hash_move_forward_ex(element_hash, &element_point), label_idx++) {

					sub_element = zend_hash_get_current_data_ex(element_hash, &element_point);

					zend_long id;
					zval *find_result;
					switch (label_idx)
					{
					case 0:
						id = Z_LVAL_P(sub_element);
						//php_printf("%d ", Z_LVAL_P(sub_element));
						find_result = zend_hash_index_find(column1_hash, id);
						row_idx = Z_LVAL_P(find_result);
						break;
					case 1:
						id = Z_LVAL_P(sub_element);
						//php_printf("%d ", Z_LVAL_P(sub_element));
						find_result = zend_hash_index_find(column2_hash, id);
						col_idx = Z_LVAL_P(find_result);
						break;
					case 2:
						value = Z_TYPE_P(sub_element) == IS_LONG ? Z_LVAL_P(sub_element) : Z_DVAL_P(sub_element);
						break;
					default:
						break;
					}
				}

				add_next_index_long(&triplet_array, row_idx);
				add_next_index_long(&triplet_array, col_idx);
				add_next_index_double(&triplet_array, value);
			}
		}


		if (valid)
		{
			//RETURN_STRING("Hello World");
			array_init(return_value);
			add_next_index_zval(return_value, &triplet_array);
			add_next_index_zval(return_value, &column1);
			add_next_index_zval(return_value, &column2);
			add_next_index_zval(return_value, &column1_inv);
			add_next_index_zval(return_value, &column2_inv);
		}
	}

	if (error_flag >= 0)
	{
		switch (error_flag)
		{
		case 0: php_printf("The input is not a array, return the error flag %d", error_flag);
			break;
		case 1: php_printf("The element of input array isn't array, return the error flag %d", error_flag);
			break;
		case 2: php_printf("The size of element array is not 3, return the error flag %d", error_flag);
			break;
		case 3: php_printf("The element keyValue is not string, return the error flag %d", error_flag);
			break;
		case 4: php_printf("The element KeyLabel is not coincident, return the error flag %d", error_flag);
			break;
		case 5: php_printf("The element key Value is not numerial(Long or Double), return the error flag %d", error_flag);
			break;
		default:
			break;
		}
		RETURN_LONG(error_flag);
	}

}*/

PHP_FUNCTION(generate_triplet_array_stupid)
{
	zval *input_table, *element;
	zend_bool bfill = 0;
	double fill_value = -1;

	if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "a|bd", &input_table, &bfill, &fill_value) == FAILURE) {
		return;
	}

	int table_count, element_count;
	HashTable *table_hash, *element_hash;
	HashPosition pointer, element_point;

	int error_flag = 0;
	if (Z_TYPE_P(input_table) == IS_ARRAY)
	{
		error_flag = -1;

		table_hash = Z_ARRVAL_P(input_table);
		table_count = zend_hash_num_elements(table_hash);

		zval column1, column2;
		array_init(&column1);
		array_init(&column2);

		int valid = 1;

		//Do some check work and get the column1 and column2
		char *label_name[3] = { NULL, NULL, NULL };
		for (zend_hash_internal_pointer_reset_ex(table_hash, &pointer);
			 zend_hash_has_more_elements_ex(table_hash, &pointer) == SUCCESS;
			 zend_hash_move_forward_ex(table_hash, &pointer)) {
			element = zend_hash_get_current_data_ex(table_hash, &pointer);

			switch (Z_TYPE_P(element))
			{
			case IS_ARRAY:
				element_hash = Z_ARRVAL_P(element);
				if (zend_hash_num_elements(element_hash) == 3)
				{
					zval *sub_element, sub_key;
					int label_idx = 0;
					for (zend_hash_internal_pointer_reset_ex(element_hash, &element_point);
						 zend_hash_has_more_elements_ex(element_hash, &element_point) == SUCCESS;
						 zend_hash_move_forward_ex(element_hash, &element_point), label_idx++) {
						zend_hash_get_current_key_zval_ex(element_hash, &sub_key, &element_point);

						if (Z_TYPE(sub_key) == IS_STRING)
						{
							char *key_str = (*(Z_STR(sub_key))).val;
							if (key_str)
							{
								sub_element = zend_hash_get_current_data_ex(element_hash, &element_point);
								//php_printf("%d </br>", Z_LVAL_P(sub_element));
								if (Z_TYPE_P(sub_element) != IS_LONG && Z_TYPE_P(sub_element) != IS_DOUBLE)
								{
									valid = 0;
									error_flag = 5;
									break;
								}

								switch (label_idx)
								{
								case 0:
									add_index_long(&column1, Z_LVAL_P(sub_element), 0);
									break;
								case 1:
									add_index_long(&column2, Z_LVAL_P(sub_element), 0);
									break;
								default:
									break;
								}
							}

							if (label_name[label_idx] && strcmp(label_name[label_idx], key_str) != 0)
							{
								valid = 0;
								error_flag = 4;
								break;
							}
							else
								label_name[label_idx] = key_str;
						}
						else
						{
							valid = 0;
							error_flag = 3;
							break;
						}
					}
				}
				else
				{
					valid = 0;
					error_flag = 2;
				}
				break;
			/*case IS_OBJECT:
				break;*/
			default:
				valid = 0;
				error_flag = 1;
				break;
			}

			if (!valid)
				break;
		}


		//Inverse column1 and column2 and set the true value of column1 and column2
		HashTable *column_hash;
		HashPosition column_pointer;
		zval column1_inv, column2_inv, key;
		int rows, cols;
		array_init(&column1_inv);
		array_init(&column2_inv);
		int inv_idx = 0;
		column_hash = Z_ARRVAL_P(&column1);
		for (zend_hash_internal_pointer_reset_ex(column_hash, &column_pointer);
			 zend_hash_has_more_elements_ex(column_hash, &column_pointer) == SUCCESS;
			 zend_hash_move_forward_ex(column_hash, &column_pointer), inv_idx++)
		{
			element = zend_hash_get_current_data_ex(column_hash, &column_pointer);
			zend_hash_get_current_key_zval_ex(column_hash, &key, &column_pointer);

			add_index_long(&column1_inv, inv_idx, Z_LVAL_P(&key));
			add_index_long(&column1, Z_LVAL_P(&key), inv_idx);
		}
		rows = inv_idx;

		inv_idx = 0;
		column_hash = Z_ARRVAL_P(&column2);
		for (zend_hash_internal_pointer_reset_ex(column_hash, &column_pointer);
			 zend_hash_has_more_elements_ex(column_hash, &column_pointer) == SUCCESS;
			 zend_hash_move_forward_ex(column_hash, &column_pointer), inv_idx++)
		{
			element = zend_hash_get_current_data_ex(column_hash, &column_pointer);
			zend_hash_get_current_key_zval_ex(column_hash, &key, &column_pointer);

			add_index_long(&column2_inv, inv_idx, Z_LVAL_P(&key));
			add_index_long(&column2, Z_LVAL_P(&key), inv_idx);
		}
		cols = inv_idx;

		//Get triplets array , the error check is not needed here
		HashTable *column1_hash, *column2_hash;
		column1_hash = Z_ARRVAL_P(&column1);
		column2_hash = Z_ARRVAL_P(&column2);
		zval triplet_array;
		array_init(&triplet_array);

		if (bfill)
		{
			int matrix_count = rows * cols;
			double *densematrix = (double *)malloc(sizeof(double) * matrix_count);
			int m_idx;
			for (m_idx = 0; m_idx < matrix_count; m_idx++)
				densematrix[m_idx] = fill_value;

			for (zend_hash_internal_pointer_reset_ex(table_hash, &pointer);
				 zend_hash_has_more_elements_ex(table_hash, &pointer) == SUCCESS;
				 zend_hash_move_forward_ex(table_hash, &pointer)) {

				element = zend_hash_get_current_data_ex(table_hash, &pointer);
				element_hash = Z_ARRVAL_P(element);

				zval *sub_element, sub_key;
				int label_idx = 0;
				int row_idx, col_idx;
				double value;
				for (zend_hash_internal_pointer_reset_ex(element_hash, &element_point);
					 zend_hash_has_more_elements_ex(element_hash, &element_point) == SUCCESS;
					 zend_hash_move_forward_ex(element_hash, &element_point), label_idx++) {

					sub_element = zend_hash_get_current_data_ex(element_hash, &element_point);

					zend_long id;
					zval *find_result;
					switch (label_idx)
					{
					case 0:
						id = Z_LVAL_P(sub_element);
						//php_printf("%d ", Z_LVAL_P(sub_element));
						find_result = zend_hash_index_find(column1_hash, id);
						row_idx = Z_LVAL_P(find_result);
						break;
					case 1:
						id = Z_LVAL_P(sub_element);
						//php_printf("%d ", Z_LVAL_P(sub_element));
						find_result = zend_hash_index_find(column2_hash, id);
						col_idx = Z_LVAL_P(find_result);
						break;
					case 2:
						value = Z_TYPE_P(sub_element) == IS_LONG ? Z_LVAL_P(sub_element) : Z_DVAL_P(sub_element);
						break;
					default:
						break;
					}
				}

				int pos = row_idx * cols + col_idx;
				densematrix[pos] = value;
			}

			int r, c;
			for (r = 0, m_idx = 0; r < rows; r++)
			{
				for (c = 0; c < cols; c++, m_idx++)
				{
					add_next_index_long(&triplet_array, r);
					add_next_index_long(&triplet_array, c);
					add_next_index_double(&triplet_array, densematrix[m_idx]);
				}
			}
		}
		else
		{
			for (zend_hash_internal_pointer_reset_ex(table_hash, &pointer);
				 zend_hash_has_more_elements_ex(table_hash, &pointer) == SUCCESS;
				 zend_hash_move_forward_ex(table_hash, &pointer)) {

				element = zend_hash_get_current_data_ex(table_hash, &pointer);
				element_hash = Z_ARRVAL_P(element);

				zval *sub_element, sub_key;
				int label_idx = 0;
				int row_idx, col_idx;
				double value;
				for (zend_hash_internal_pointer_reset_ex(element_hash, &element_point);
					 zend_hash_has_more_elements_ex(element_hash, &element_point) == SUCCESS;
					 zend_hash_move_forward_ex(element_hash, &element_point), label_idx++) {

					sub_element = zend_hash_get_current_data_ex(element_hash, &element_point);

					zend_long id;
					zval *find_result;
					switch (label_idx)
					{
					case 0:
						id = Z_LVAL_P(sub_element);
						//php_printf("%d ", Z_LVAL_P(sub_element));
						find_result = zend_hash_index_find(column1_hash, id);
						row_idx = Z_LVAL_P(find_result);
						break;
					case 1:
						id = Z_LVAL_P(sub_element);
						//php_printf("%d ", Z_LVAL_P(sub_element));
						find_result = zend_hash_index_find(column2_hash, id);
						col_idx = Z_LVAL_P(find_result);
						break;
					case 2:
						value = Z_TYPE_P(sub_element) == IS_LONG ? Z_LVAL_P(sub_element) : Z_DVAL_P(sub_element);
						break;
					default:
						break;
					}
				}

				add_next_index_long(&triplet_array, row_idx);
				add_next_index_long(&triplet_array, col_idx);
				add_next_index_double(&triplet_array, value);
			}
		}


		if (valid)
		{
			//RETURN_STRING("Hello World");
			array_init(return_value);
			add_next_index_zval(return_value, &triplet_array);
			add_next_index_zval(return_value, &column1);
			add_next_index_zval(return_value, &column2);
			add_next_index_zval(return_value, &column1_inv);
			add_next_index_zval(return_value, &column2_inv);
		}
	}

	if (error_flag >= 0)
	{
		switch (error_flag)
		{
		case 0: php_printf("The input is not a array, return the error flag %d", error_flag);
			break;
		case 1: php_printf("The element of input array isn't array, return the error flag %d", error_flag);
			break;
		case 2: php_printf("The size of element array is not 3, return the error flag %d", error_flag);
			break;
		case 3: php_printf("The element keyValue is not string, return the error flag %d", error_flag);
			break;
		case 4: php_printf("The element KeyLabel is not coincident, return the error flag %d", error_flag);
			break;
		case 5: php_printf("The element key Value is not numerial(Long or Double), return the error flag %d", error_flag);
			break;
		default:
			break;
		}
		RETURN_LONG(error_flag);
	}

}

PHP_FUNCTION(generate_triplet_array)
{
	zval *input_table, *column1, *column2, *element;
	zend_bool bfill = 0;
	double fill_value = -1;

	if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "aaa|bd", &input_table, &column1, &column2, &bfill, &fill_value) == FAILURE) {
		return;
	}

	int table_count, element_count;
	HashTable *table_hash, *element_hash;
	HashPosition pointer, element_point;

	int error_flag = 0;

	if (Z_TYPE_P(input_table) == IS_ARRAY &&
		Z_TYPE_P(column1) == IS_ARRAY &&
		Z_TYPE_P(column2) == IS_ARRAY)
	{
		error_flag = -1;

		table_hash = Z_ARRVAL_P(input_table);
		table_count = zend_hash_num_elements(table_hash);

		//Inverse column1 column2 and do some check for column1 column2
		int valid = 1;
		HashTable *column_hash;
		HashPosition column_pointer;
		zval column1_inv, column2_inv, key;
		int rows = 0, cols = 0;
		array_init(&column1_inv);
		array_init(&column2_inv);
		int inv_idx = 0;
		column_hash = Z_ARRVAL_P(column1);
		for (zend_hash_internal_pointer_reset_ex(column_hash, &column_pointer);
			 zend_hash_has_more_elements_ex(column_hash, &column_pointer) == SUCCESS;
			 zend_hash_move_forward_ex(column_hash, &column_pointer), inv_idx++)
		{
			element = zend_hash_get_current_data_ex(column_hash, &column_pointer);
			if (Z_TYPE_P(element) != IS_LONG)
			{
				valid = 0;
				error_flag = 1;
				break;
			}
			add_index_long(&column1_inv, Z_LVAL_P(element), inv_idx);
		}
		rows = inv_idx;

		if (valid)
		{
			inv_idx = 0;
			column_hash = Z_ARRVAL_P(column2);
			for (zend_hash_internal_pointer_reset_ex(column_hash, &column_pointer);
				 zend_hash_has_more_elements_ex(column_hash, &column_pointer) == SUCCESS;
				 zend_hash_move_forward_ex(column_hash, &column_pointer), inv_idx++)
			{
				element = zend_hash_get_current_data_ex(column_hash, &column_pointer);
				if (Z_TYPE_P(element) != IS_LONG)
				{
					valid = 0;
					error_flag = 2;
					break;
				}
				add_index_long(&column2_inv, Z_LVAL_P(element), inv_idx);
			}
			cols = inv_idx;

			if (valid)
			{
				//Get triplets array , the error check is not needed here
				HashTable *column1_inv_hash, *column2_inv_hash;
				column1_inv_hash = Z_ARRVAL_P(&column1_inv);
				column2_inv_hash = Z_ARRVAL_P(&column2_inv);
				zval triplet_array;
				array_init(&triplet_array);

				int matrix_count = rows * cols;
				double *densematrix = NULL;
				if (bfill)
				{
					densematrix = (double *)malloc(sizeof(double) * matrix_count);
					int m_idx;
					for (m_idx = 0; m_idx < matrix_count; m_idx++)
						densematrix[m_idx] = fill_value;
				}

				char *label_name[3] = { NULL, NULL, NULL };
				for (zend_hash_internal_pointer_reset_ex(table_hash, &pointer);
					 zend_hash_has_more_elements_ex(table_hash, &pointer) == SUCCESS;
					 zend_hash_move_forward_ex(table_hash, &pointer)) {
					element = zend_hash_get_current_data_ex(table_hash, &pointer);

					switch (Z_TYPE_P(element))
					{
					case IS_ARRAY:
						element_hash = Z_ARRVAL_P(element);
						if (zend_hash_num_elements(element_hash) == 3)
						{
							zval *sub_element, sub_key;
							int label_idx = 0, row_idx = 0, col_idx = 0;
							double value = 0;
							for (zend_hash_internal_pointer_reset_ex(element_hash, &element_point);
								 zend_hash_has_more_elements_ex(element_hash, &element_point) == SUCCESS;
								 zend_hash_move_forward_ex(element_hash, &element_point), label_idx++) {
								zend_hash_get_current_key_zval_ex(element_hash, &sub_key, &element_point);

								if (Z_TYPE(sub_key) == IS_STRING)
								{
									char *key_str = (*(Z_STR(sub_key))).val;
									sub_element = zend_hash_get_current_data_ex(element_hash, &element_point);

									zval *find_result;
									switch (label_idx)
									{
									case 0:
										if (Z_TYPE_P(sub_element) == IS_LONG)
										{
											find_result = zend_hash_index_find(column1_inv_hash, Z_LVAL_P(sub_element));
											row_idx = Z_LVAL_P(find_result);
										}
										else
										{
											valid = 0;
											error_flag = 3;
										}
										break;
									case 1:
										if (Z_TYPE_P(sub_element) == IS_LONG)
										{
											find_result = zend_hash_index_find(column2_inv_hash, Z_LVAL_P(sub_element));
											col_idx = Z_LVAL_P(find_result);
										}
										else
										{
											valid = 0;
											error_flag = 3;
										}
										break;
									case 2:
										if (Z_TYPE_P(sub_element) == IS_LONG || Z_TYPE_P(sub_element) == IS_DOUBLE)
										{
											value = Z_TYPE_P(sub_element) == IS_LONG ? Z_LVAL_P(sub_element) : Z_DVAL_P(sub_element);
										}
										else
										{
											valid = 0;
											error_flag = 3;
										}
										break;
									default:
										break;
									}

									if (!valid)break;

									if (label_name[label_idx] && strcmp(label_name[label_idx], key_str) != 0)
									{
										valid = 0;
										error_flag = 4;
										break;
									}
									else
										label_name[label_idx] = key_str;
								}
								else
								{
									valid = 0;
									error_flag = 5;
									break;
								}
							}

							if (bfill)
							{
								int pos = row_idx * cols + col_idx;
								densematrix[pos] = value;
							}
							else
							{
								add_next_index_long(&triplet_array, row_idx);
								add_next_index_long(&triplet_array, col_idx);
								add_next_index_double(&triplet_array, value);
							}
						}
						else
						{
							valid = 0;
							error_flag = 6;
						}
						break;
					default:
						valid = 0;
						error_flag = 6;
						break;
					}

					if (!valid)
						break;
				}

				if (bfill && valid)
				{

					int r, c, m_idx;
					for (r = 0, m_idx = 0; r < rows; r++)
					{
						for (c = 0; c < cols; c++, m_idx++)
						{
							add_next_index_long(&triplet_array, r);
							add_next_index_long(&triplet_array, c);
							add_next_index_double(&triplet_array, densematrix[m_idx]);
						}
					}

					free(densematrix);
				}

				if (valid)
				{
					array_init(return_value);
					ZVAL_COPY(return_value, &triplet_array);
				}
			}
		}
	}

	if (error_flag >= 0)
	{
		switch (error_flag)
		{
		case 0: php_printf("The input is not three arraies, return the error flag %d", error_flag);
			break;
		case 1: php_printf("The element of column1 must the Long integer, return the error flag %d", error_flag);
			break;
		case 2: php_printf("The element of column2 must the Long integer, return the error flag %d", error_flag);
			break;
		case 3: php_printf("The value of labels is not numerical, return the error flag %d", error_flag);
			break;
		case 4: php_printf("The element KeyLabel is not coincident, return the error flag %d", error_flag);
			break;
		case 5: php_printf("The element KeyLabel is not string type, return the error flag %d", error_flag);
			break;
		case 6: php_printf("The element of input table is not a array with 3 elements, return the error flag %d", error_flag);
			break;
		default:
			break;
		}
		RETURN_LONG(error_flag);
	}

}


inline float * farray_from_zendarray(zval *zarray, int *len)
{
  if(Z_TYPE_P(zarray) != IS_ARRAY)
    return NULL;
  HashTable *arr_hash = Z_ARRVAL_P(zarray);
  HashPosition pointer;
  int arr_lens = zend_hash_num_elements(arr_hash);

  float *result = (float *)malloc(arr_lens * sizeof(float));
  int resultIdx = 0, valid = 1;
  zval *element;
  for (zend_hash_internal_pointer_reset_ex(arr_hash, &pointer);
  zend_hash_has_more_elements_ex(arr_hash, &pointer) == SUCCESS;
  zend_hash_move_forward_ex(arr_hash, &pointer), resultIdx++)
  {
    element = zend_hash_get_current_data_ex(arr_hash, &pointer);
    if(Z_TYPE_P(element) == IS_LONG || Z_TYPE_P(element) == IS_DOUBLE)
    {
      result[resultIdx] = Z_TYPE_P(element) == IS_LONG ? (float)Z_LVAL_P(element) : (float)Z_DVAL_P(element);
      //php_printf(" %d %f</br>",resultIdx,result[resultIdx == 0? 0: resultIdx -1]);
    }
    else{
      //php_printf(" Bug");
      valid = 0;
      break;
    }
  }


  if(valid == 0)
  {
    free(result);
    *len = 0;
    result = NULL;
  }
  else{
    *len = arr_lens;
  }

  return result;
}

/* {{{ PHP_INI
 */
/* Remove comments and fill if you need to have entries in php.ini
PHP_INI_BEGIN()
    STD_PHP_INI_ENTRY("php_libmf.global_value",      "42", PHP_INI_ALL, OnUpdateLong, global_value, zend_php_libmf_globals, php_libmf_globals)
    STD_PHP_INI_ENTRY("php_libmf.global_string", "foobar", PHP_INI_ALL, OnUpdateString, global_string, zend_php_libmf_globals, php_libmf_globals)
PHP_INI_END()
*/
/* }}} */

/* Remove the following function when you have successfully modified config.m4
   so that your module can be compiled into PHP, it exists only for testing
   purposes. */

/* Every user-visible function in PHP should document itself in the source */
/* {{{ proto string confirm_php_libmf_compiled(string arg)
   Return a string to confirm that the module is compiled in */

PHP_FUNCTION(mf_my_train)
{
    char *tr_path = "";
    char *model_path = "";
	size_t tr_path_len, model_path_len;

	if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "ss", &tr_path, &tr_path_len, &model_path, &model_path_len) == FAILURE) {
		return;
	}
    int result = php_mf_my_train(tr_path, model_path);

    switch (result)
   {
	case 0:
		php_printf(" train successfully \n");
		break;
	case -1:
		php_printf(" train failed \n");
		break;
	default:
		break;
  }

  RETURN_LONG(result);
}

PHP_FUNCTION(utility_train)
{
      zval *triplet_arr;
      double p_l2 = 0.1;
      double q_l2 = 0.1;
      int k = 8;
      int iters = 30;
      double eta = 0.1;

      if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "addlld", &triplet_arr, &p_l2,&q_l2,&k,&iters,&eta) == FAILURE) {
		return;
	}
      int errorFlag = 0;

      int arr_count, triplet_count;
      float *ftriplet_arr = farray_from_zendarray(triplet_arr, &arr_count);
      if(ftriplet_arr != NULL)
      {
        if(arr_count % 3 == 0)
        {
          errorFlag = -1;
          triplet_count = arr_count / 3;
          //int i  = 0;
          //for( ; i < triplet_count; i++)
          //{
          //  php_printf("Train data: %f,%f,%f </br>",ftriplet_arr[i*3],ftriplet_arr[i*3+1],ftriplet_arr[i*3+2]);
          //}
          int result_lens = 0;

          float * train_result = php_utility_train(ftriplet_arr,triplet_count,p_l2, q_l2, k, iters, eta, &result_lens);
          //php_printf("model_parameters: %d ,%f,%f,%d,%d,%f,%d</br>",triplet_count,p_l2, q_l2, k, iters, eta, result_lens);
          //RETURN_STRING("Hello Bug");
          //char *teststr = "Hello Bug";
          //php_printf("%d triplet_count = %d", strlen(train_result), triplet_count);

          //php_printf("canshu: %f %f %d %d %f \n",p_l2, q_l2, k, iters, eta);
          //php_printf(" strlen(train_result) = %d \n",strlen(train_result));
          //int v = 0;
          //for (; v < result_lens; v++) {
          //  php_printf("index,model_arr:  %d ,%f</br>",v,train_result[v]);
          //}
          array_init(return_value);
          int k = 0,j = 0;
          for(; k < result_lens; k++, j ++)
          {
              add_index_double(return_value, k, train_result[k]);
          }

          //RETURN_STRING(train_result);
        }
        else
          errorFlag = 1;

      }

      switch(errorFlag)
      {
        case 0:
           php_printf(" The input is not a array , Or that the input array is not whole numerical \n");
	          break;
       case 1:
            php_printf(" The size of input array is not the times of three \n");
            break;

       default:
	     break;
      }

     if(errorFlag >= 0) RETURN_LONG(errorFlag);
}

PHP_FUNCTION(utility_predict)
{
   zval *test_arr,*model_arr;

   if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "aa", &test_arr, &model_arr) == FAILURE) {
		   return;
	}
   int errorFlag = 0;

   int test_arr_count,test_triplet_count;
   float *ftest_arr = farray_from_zendarray(test_arr, &test_arr_count);
   //php_printf("test_arr_count: %d </br>",test_arr_count);
   int model_arr_count;
   float *fmodel_arr = farray_from_zendarray(model_arr, &model_arr_count);

   if(ftest_arr != NULL&&fmodel_arr!= NULL)
   {
     if(test_arr_count % 2 == 0)
     {
       errorFlag = -1;
       test_triplet_count = test_arr_count / 2;
       //int i  = 0;
       //for( ; i < test_triplet_count; i++)
       //{
       //  php_printf("Test data: %f,%f </br>",ftest_arr[i*2],ftest_arr[i*2+1]);
       //}
       float * predict_v = php_utility_predict(ftest_arr, test_triplet_count,fmodel_arr,model_arr_count);
       array_init(return_value);
       int k = 0,j = 0;
       for(; k < test_triplet_count; k++, j ++)
       {
           add_index_double(return_value, k, predict_v[k]);
       }
     }
     else
       errorFlag = 1;
   }
   switch(errorFlag)
   {
     case 0:
        php_printf(" The input is not a array , Or that the input array is not whole numerical \n");
         break;
    case 1:
         php_printf(" The size of input array is not the times of two \n");
         break;

    default:
    break;
   }

}

PHP_FUNCTION(cos_similarity)
{
    zval* triplet_arr, *data;
    HashTable *arr_hash;
    HashPosition pointer;
    int q_rows,q_col;
    int triplet_num, item_id;
    if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "lall",&item_id,&triplet_arr,&q_rows, &q_col) == FAILURE)
       {
         return;
		   }

    int errorFlag = 0;
    if (Z_TYPE_P(triplet_arr) == IS_ARRAY)
  	{
  		  arr_hash = Z_ARRVAL_P(triplet_arr);
  		  triplet_num = zend_hash_num_elements(arr_hash) / 3;

        int valid = 1;

        if(q_rows * q_col  == triplet_num )
        {
            float *result = (float *)malloc(triplet_num * 3 * sizeof(float));
    		    int resultIdx = 0;
    		    for (zend_hash_internal_pointer_reset_ex(arr_hash, &pointer);zend_hash_has_more_elements_ex(arr_hash, &pointer) == SUCCESS; zend_hash_move_forward_ex(arr_hash, &pointer))
            {
    			      data = zend_hash_get_current_data_ex(arr_hash, &pointer);
    			      switch (Z_TYPE_P(data))
    			      {
    			         case IS_LONG:
    				            result[resultIdx] = (float)Z_LVAL_P(data);
    				       break;
    			         case IS_DOUBLE:
    				            result[resultIdx] = (float)Z_DVAL_P(data);
    				       break;
    			         default:
    				            valid = 0;
    				       break;
    			      }

              //php_printf(" %d %f</br>",resultIdx,result[resultIdx]);

    			    if (valid == 0)
              {
                  errorFlag = 1;
                  break;
              }

    			    resultIdx++;
    		    }

    		    if (valid != 0)
    		    {
               /*php_printf("item_id %d </br>",item_id);
               int k = 0;
               for (; k < triplet_num; k++)
               {

                  php_printf("%f %f %f </br>",result[k*3],result[k*3+1],result[k*3+2]);
               }
               php_printf("triplet_num %d </br>",triplet_num);*/
    			     float * cos_arr = php_cos_similarity(item_id,result,triplet_num);
               //php_printf("cos_arr %f </br>",cos_arr);

               array_init(return_value);
               int i = 0, j = 0;
               for(; i < q_rows; i++, j += 2)
               {
                   add_index_double(return_value, i, cos_arr[i]);
               }

               //RETURN_STRING("Hello");
              errorFlag = -1;
    		    }
            free(result);
       }
       else
       {
          valid = 0;
          errorFlag = 2;
       }
  	}

    switch(errorFlag)
    {
       case 0:
       php_printf("no array \n");
       break;
       case 1:
       php_printf(" no numerical!  \n");
       case 2:
       php_printf(" The dimensions of matrix is not coincides to triplets  \n");
       break;
       default:
       break;
    }
    if(errorFlag != -1)RETURN_LONG(errorFlag);
}

PHP_FUNCTION(DINA)
{
    zval* q_arr, *x_arr;
    int iters;
    if (zend_parse_parameters(ZEND_NUM_ARGS() TSRMLS_CC, "aal", &q_arr, &x_arr,&iters) == FAILURE) {
		return;
	}
    int errorFlag = 0;
    int q_arr_count,q_triplet_num;
    int x_arr_count,x_triplet_num;
    float *fq_arr = farray_from_zendarray(q_arr, &q_arr_count);
    float *fx_arr = farray_from_zendarray(x_arr, &x_arr_count);
     if(fq_arr != NULL&&fx_arr!= NULL)
     {
       if(q_arr_count % 3 == 0&&x_arr_count % 3 == 0)
       {
         errorFlag = -1;
         q_triplet_num = q_arr_count/3;
         x_triplet_num = x_arr_count/3;
         //check the input q_arr and x_arr
         //int i = 0,j = 0;
         //for( ; i < q_triplet_num; i++)
         //{
           //php_printf("fq_arr data: %f, %f ,%f</br>",fq_arr[i*3],fq_arr[i*3+1],fq_arr[i*3+2]);
         //}
         //for( ; j < x_triplet_num; j++)
         //{
          // php_printf("fx_arr data: %f, %f ,%f</br>",fx_arr[j*3],fx_arr[j*3+1],fx_arr[j*3+2]);
         //}

         //php_printf("DINA_parameters: %d ,%d,%d</br>",q_triplet_num,x_triplet_num,iters);
         int * dina_res = php_DINA(fq_arr,q_triplet_num,fx_arr,x_triplet_num,iters);

         //RETURN_STRING("Hello Bug");
         array_init(return_value);
         int k = 0,m = 0;
         //int predict_v_len = x_triplet_num*
         for(; k < 20; k++,m+=2)
         {
             add_index_double(return_value, k, dina_res[k]);
         }

       }
       else
       {errorFlag = 1;}
     }
     //error check
     switch(errorFlag)
     {
       case 0:
          php_printf(" The input is not a array , Or that the input array is not whole numerical \n");
           break;
      case 1:
           php_printf(" The size of input array is not the times of two \n");
           break;

      default:
      break;
     }


}


/* }}} */
/* The previous line is meant for vim and emacs, so it can correctly fold and
   unfold functions in source code. See the corresponding marks just before
   function definition, where the functions purpose is also documented. Please
   follow this convention for the convenience of others editing your code.
*/


/* {{{ php_php_libmf_init_globals
 */
/* Uncomment this function if you have INI entries
static void php_php_libmf_init_globals(zend_php_libmf_globals *php_libmf_globals)
{
	php_libmf_globals->global_value = 0;
	php_libmf_globals->global_string = NULL;
}
*/
/* }}} */

/* {{{ PHP_MINIT_FUNCTION
 */
PHP_MINIT_FUNCTION(mf)
{
	/* If you have INI entries, uncomment these lines
	REGISTER_INI_ENTRIES();
	*/
	return SUCCESS;
}
/* }}} */

/* {{{ PHP_MSHUTDOWN_FUNCTION
 */
PHP_MSHUTDOWN_FUNCTION(mf)
{
	/* uncomment this line if you have INI entries
	UNREGISTER_INI_ENTRIES();
	*/
	return SUCCESS;
}
/* }}} */

/* Remove if there's nothing to do at request start */
/* {{{ PHP_RINIT_FUNCTION
 */
PHP_RINIT_FUNCTION(mf)
{
	return SUCCESS;
}
/* }}} */

/* Remove if there's nothing to do at request end */
/* {{{ PHP_RSHUTDOWN_FUNCTION
 */
PHP_RSHUTDOWN_FUNCTION(mf)
{
	return SUCCESS;
}
/* }}} */

/* {{{ PHP_MINFO_FUNCTION
 */
PHP_MINFO_FUNCTION(mf)
{
	php_info_print_table_start();
	php_info_print_table_header(2, "mf support", "enabled");
	php_info_print_table_end();

	/* Remove comments if you have entries in php.ini
	DISPLAY_INI_ENTRIES();
	*/
}
/* }}} */

/* {{{ php_libmf_functions[]
 *
 * Every user visible function must have an entry in php_libmf_functions[].
 */
const zend_function_entry mf_functions[] = {
	      PHP_FE(mf_my_train,	NULL)		/* For testing, remove later. */
        PHP_FE(generate_triplet_array,	NULL)
        PHP_FE(generate_triplet_array_stupid,	NULL)
        PHP_FE(utility_train,	NULL)
        PHP_FE(utility_predict,	NULL)
        PHP_FE(cos_similarity,	NULL)
        PHP_FE(DINA,	        NULL)
	PHP_FE_END	/* Must be the last line in php_libmf_functions[] */
};
/* }}} */

/* {{{ php_libmf_module_entry
 */
zend_module_entry mf_module_entry = {
	STANDARD_MODULE_HEADER,
	"mf",
	mf_functions,
	PHP_MINIT(mf),
	PHP_MSHUTDOWN(mf),
	PHP_RINIT(mf),		/* Replace with NULL if there's nothing to do at request start */
	PHP_RSHUTDOWN(mf),	/* Replace with NULL if there's nothing to do at request end */
	PHP_MINFO(mf),
	PHP_MF_VERSION,
	STANDARD_MODULE_PROPERTIES
};
/* }}} */

#ifdef COMPILE_DL_MF
ZEND_GET_MODULE(mf)
#endif

/*
 * Local variables:
 * tab-width: 4
 * c-basic-offset: 4
 * End:
 * vim600: noet sw=4 ts=4 fdm=marker
 * vim<600: noet sw=4 ts=4
 */
