#include <assert.h>
#include <time.h>
#include <stdbool.h>
#include <unistd.h>
#include "minunit.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_sf.h>
#include "../src/semanticHash/dbg.h"
#include "../src/semanticHash/rmb.h"

#define EPSILON 0.0000001 // for float_check

int float_check(double a, double b);
char * test_layer_creation();
char * test_hidden_probablities_blas();
char * test_hidden_probablities_on_data();
char * test_visible_probablities_blas();
char * test_visible_probablities_on_data();
char * test_pretraining();

/* Helpers */

int float_check(double a, double b)
{
	return (a - b) < EPSILON;
}

void setup_layer(gsl_matrix * input, gsl_matrix * weights, gsl_vector * hidden_bias, gsl_vector * visible_bias)
{
	unsigned int  j, k;
	gsl_rng * randgen;
	gsl_matrix * new_hidden;
	randgen = gsl_rng_alloc (gsl_rng_taus); //fastest
	gsl_rng_set(randgen, (unsigned long int) 10);//time(NULL));
	for(j = 0; j < weights->size1; j++) {
		for(k = 0; k < weights->size2; k++) {
			gsl_matrix_set(weights, j, k, gsl_ran_gaussian(randgen, 0.01));
		}
	}
	for(j = 0; j < input->size1; j++) {
		for(k = 0; k < input->size2; k++) {
			gsl_matrix_set(input, j, k, (double)gsl_rng_uniform_int(randgen, 100));
		}
	}
	gsl_vector_set_all(hidden_bias, 1);
	gsl_vector_set_all(visible_bias, 0.4);
	gsl_rng_free(randgen);
	// [m,z]

}

/* Tests */
char * test_layer_creation()
{
	int dim[] = {3, 4, 2};
	PretrainModel * rbm = pretrain_create(0.1, dim);
	mu_assert(rbm->first_layer != NULL, "didnot create initial layer");
	mu_assert(rbm->last_layer != NULL, "didnot set last layer");
	mu_assert(rbm->learning_rate == 0.1, "error rate not set");
	mu_assert(rbm->first_layer->dimension == 3, " initial layer dimension not set properly");
	mu_assert(rbm->first_layer->next->dimension == 4, "layer dimension not set properly");
	mu_assert(rbm->first_layer->next->next->dimension == 2, "layer dimension not set properly");
	mu_assert(rbm->first_layer->next->next->next == NULL, "layer");
	mu_assert(rbm->last_layer == rbm->first_layer->next->next, "layer");
    mu_assert(rbm->first_layer->weights->size1 == 3,"first weight rows error");
    mu_assert(rbm->first_layer->weights->size2 == 4,"weight columns error");
    mu_assert(rbm->first_layer->next->weights->size1 == 3,"weight layer improper");
    mu_assert(rbm->first_layer->next->weights->size2 == 4,"weight layer improper");
    mu_assert(rbm->first_layer->next->next->weights->size1 == 4,"weight layer improper");
    mu_assert(rbm->first_layer->next->next->weights->size2 == 2,"weight layer improper");
	log_success("Creation of layers");
	return NULL;
}

char * test_hidden_probablities_blas()
{

	// [m,n] matrix, [1,z] vector , [n,z] matrix
	unsigned int m = 10;
	unsigned int n = 200;
	unsigned int z = 50;
	gsl_matrix * test_visible = gsl_matrix_alloc(m, n); // row major
	gsl_matrix * test_weights = gsl_matrix_alloc(n, z);  //col major
	gsl_vector * test_hidden_bias = gsl_vector_calloc(z);

	gsl_matrix * new_hidden = get_batched_hidden_probablities_blas(test_visible, test_hidden_bias, test_weights);
	mu_assert(new_hidden->size1 == test_visible->size1, "Mismatched in to output row lengths");
	mu_assert(new_hidden->size2 == test_hidden_bias->size, "Mismatched column lengths");
	mu_assert(new_hidden->size2 != test_visible->size2, "Mismatched column lengths");
	mu_assert(new_hidden->size1 == m, "Mismatched in to output row lengths");
	mu_assert(new_hidden->size2 == z, "Mismatched in to output column lengths");

	log_success("hidden probablity works");
    gsl_matrix_free(test_visible);
    gsl_matrix_free(test_weights);
    gsl_vector_free(test_hidden_bias);
	return NULL;
}

char * test_hidden_probablities_on_data()
{
	// SETUP
	// [m,n] matrix, [m,z] matrix, [1,z] vector, [1,n] vector, [n,z] matrix
	int m = 100;
	int n = 300;
	int z = 50;
	unsigned int i, j, k;
	gsl_matrix * weights;
	gsl_rng * randgen;
	gsl_matrix * input;
	gsl_vector * hidden_bias;
	gsl_vector * visible_bias;
	gsl_matrix * new_hidden;


	weights = gsl_matrix_alloc(n, z);
	input = gsl_matrix_alloc(m, n);
	hidden_bias = gsl_vector_alloc(z);
	visible_bias = gsl_vector_alloc(n);

	setup_layer(input, weights, hidden_bias, visible_bias);
	new_hidden = get_batched_hidden_probablities_blas(input, hidden_bias, weights);

	// TESTING
	check_hard(input->size1, "error");
	for(k = 0; k < input->size1; k++) { //each input [m,n]
		for(j = 0; j < hidden_bias->size; j++) { // bias [1,z] for each dimension of input
			double sum = 0;
			for(i = 0; i < weights->size1; i++) {
				sum += gsl_matrix_get(weights, i, j) * gsl_matrix_get(input, k, i);
			}
			double partial = sum + gsl_vector_get(hidden_bias, j);
			double result = sigmoid(partial);
			mu_assert(float_check(result , gsl_matrix_get(new_hidden, k, j)), "error in calc");
			//         log_warn("result is: %f", result);
		}
	}
	log_success("Hidden probablity works");
	gsl_vector_free(hidden_bias);
	gsl_vector_free(visible_bias);
	gsl_matrix_free(weights);
	gsl_matrix_free(input);
	gsl_matrix_free(new_hidden);
	return NULL;

}

char * test_visible_probablities_blas()
{
	gsl_matrix * test_visible = gsl_matrix_calloc(2, 4); // row major
	gsl_matrix * test_weights = gsl_matrix_calloc(4, 3);  //col major
	gsl_vector * test_hidden_bias = gsl_vector_calloc(3);
	gsl_vector * test_visible_bias = gsl_vector_calloc(4);
	gsl_matrix_set_all(test_visible, 1);
	gsl_matrix_set_all(test_weights, 1);

	gsl_matrix * new_hidden = get_batched_hidden_probablities_blas(test_visible, test_hidden_bias, test_weights);
	gsl_matrix * new_visible = get_batched_visible_probablities_blas(test_visible, new_hidden, test_visible_bias, test_weights);
	mu_assert(new_visible->size1 == test_visible->size1, "Mismatched sizes");
	log_warn("Small number of test cases");
	log_success("Visible probablity works");
	gsl_vector_free(test_hidden_bias);
	gsl_vector_free(test_visible_bias);
	gsl_matrix_free(test_weights);
	gsl_matrix_free(test_visible);
	return NULL;
}

char * test_visible_probablities_on_data()
{
	// SETUP
	// [m,n] matrix, [m,z] matrix, [1,z] vector, [1,n] vector, [n,z] matrix
	int m = 50;
	int n = 30;
	int z = 20;
	unsigned int i, j, k, v;
	gsl_matrix * weights;
	gsl_matrix * input;
	gsl_vector * hidden_bias;
	gsl_vector * visible_bias;
	gsl_matrix * new_hidden;
	gsl_matrix * reconstruction;


	weights = gsl_matrix_alloc(n, z);
	input = gsl_matrix_alloc(m, n);
	hidden_bias = gsl_vector_alloc(z);
	visible_bias = gsl_vector_alloc(n);

	setup_layer(input, weights, hidden_bias, visible_bias);
	new_hidden = get_batched_hidden_probablities_blas(input, hidden_bias, weights);
	reconstruction = get_batched_visible_probablities_blas(input, new_hidden, visible_bias, weights);
	check_hard(reconstruction->size1 == m, "error row length");
	check_hard(reconstruction->size2 == n, "error col length");
	double multiplier;
	double numerator_sum;
	double sum_result;
	double denominator_sum;
	double denom_result;

	// TESTING
	// AKA Be as verbose as possible to avoid any confusion in algorithm loops
	check_hard(new_hidden->size1 == m, "wrong Dimensions");
	check_hard(new_hidden->size1 == input->size1, "wrong Dimensions");
	check_hard(new_hidden->size2 == hidden_bias->size, "wrong Dimensions");
	check_hard(visible_bias->size == input->size2, "wrong Dimensions");
	for(k = 0; k < input->size1; k++) { //each input [m,n]
		for(i = 0; i < input->size2; i++) { // for each row dimension of input
			multiplier = 0;
			sum_result = 0;
			for(j = 0; j < input->size2; j++) {  // for each row dimension of hidden
				multiplier += gsl_matrix_get(input, k, j);
			}
			denom_result = 0;
			for(v = 0; v < visible_bias->size; v++) { // for each dimension of visible input
				denominator_sum = 0;
				for(j = 0; j < new_hidden->size2; j++) {  // for each row dimension of hidden
					denominator_sum +=  gsl_matrix_get(new_hidden, k, j) * gsl_matrix_get(weights, v, j);
				}
				denominator_sum += gsl_vector_get(visible_bias, v);
				denom_result += gsl_sf_exp(denominator_sum);
			}
			numerator_sum = 0;
			for(j = 0; j < new_hidden->size2; j++) {
				numerator_sum += gsl_matrix_get(weights, i, j) * gsl_matrix_get(new_hidden, k, j);
			}
			numerator_sum += gsl_vector_get(visible_bias, i);
			sum_result = gsl_sf_exp(numerator_sum);
			//    log_warn("test num is: %g",sum_result);
			//   log_warn("test denom is: %g",denom_result);
			//  log_warn("test multi is: %g",multiplier);
			double result = gsl_ran_poisson_pdf(gsl_matrix_get(input, k, i), (sum_result / denom_result) * multiplier);
			//log_warn("result %f", result);
			mu_assert(float_check(result , gsl_matrix_get(reconstruction, k, i)), "error in calc");
			//log_warn("recon  %f", gsl_matrix_get(reconstruction,k,i));;
			//log_warn("");
		}
	}
	log_success("visible probablity works");
	gsl_vector_free(hidden_bias);
	gsl_vector_free(visible_bias);
	gsl_matrix_free(weights);
	gsl_matrix_free(input);
	gsl_matrix_free(new_hidden);
	return NULL;

}

char * test_batch_read_in()
{
    char cwd[1024];
       if (getcwd(cwd, sizeof(cwd)) != NULL)
           log_info( "Current working dir: %s", cwd);
       else
           perror("getcwd() error");

    int batch_size = 2;
    FILE * test_file = fopen("tests/example.txt", "r");
    gsl_matrix * test_matrix = gsl_matrix_calloc(batch_size,5);
    mu_assert(test_file,"file for testing error");
    mu_assert(2==read_in_batch(test_matrix, test_file,batch_size,5),"wrong read in size");
    mu_assert(1==read_in_batch(test_matrix, test_file,batch_size,5),"wrong read in size");
    return NULL;
}

char * test_pretraining()
{
	
	// SETUP
	// [m,n] matrix, [m,z] matrix, [1,z] vector, [1,n] vector, [n,z] matrix
	int m = 1;
	int n = 2;
	int z = 2;
    double learning_rate = 0.1;
	unsigned int i, j, k, v;
	gsl_matrix * weights;
	gsl_matrix * input;
	gsl_vector * hidden_bias;
	gsl_vector * visible_bias;
	gsl_matrix * new_hidden;
	gsl_matrix * reconstruction;


	weights = gsl_matrix_alloc(n, z);
	input = gsl_matrix_alloc(m, n);
	hidden_bias = gsl_vector_alloc(z);
	visible_bias = gsl_vector_alloc(n);

    
	setup_layer(input, weights, hidden_bias, visible_bias);

    for(i=0;i<weights->size1;i++){
        for(j=0;j<weights->size2;j++){
            //log_info("new %f", gsl_matrix_get(weights,i,j));
        }
    }
	new_hidden = get_batched_hidden_probablities_blas(input, hidden_bias, weights);
	reconstruction = get_batched_visible_probablities_blas(input, new_hidden, visible_bias, weights);
    single_step_constrastive_convergence(input,visible_bias,hidden_bias,weights,learning_rate,FIRSTLAYER);
    for(i=0;i<weights->size1;i++){
        for(j=0;j<weights->size2;j++){
            //log_info("old %f", gsl_matrix_get(weights,i,j));
        }
    }
    

	log_warn("Small number of test cases");
	log_success("pretraining works");

	gsl_vector_free(hidden_bias);
	gsl_vector_free(visible_bias);
	gsl_matrix_free(weights);
	gsl_matrix_free(input);
	gsl_matrix_free(new_hidden);
	return NULL;
}

char *all_tests()
{
	mu_suite_start();
	mu_run_test(test_layer_creation);
	mu_run_test(test_hidden_probablities_blas);
	mu_run_test(test_visible_probablities_blas);
	mu_run_test(test_hidden_probablities_on_data);
	mu_run_test(test_visible_probablities_on_data);
    mu_run_test(test_batch_read_in);
	//mu_run_test(test_pretraining);
	return NULL;
}

RUN_TESTS(all_tests);


