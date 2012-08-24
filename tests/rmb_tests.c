#include <assert.h>
#include <time.h>
#include <stdbool.h>
#include "minunit.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "../src/semanticHash/dbg.h"
#include "../src/semanticHash/rmb.h"

#define EPSILON 0.001

int float_check(double a, double b)
{
    return (a - b) < EPSILON;
}

char * test_layer_creation()
{
	int dim[] = { 3, 4, 2};
	PretrainModel * rbm = pretrain_create(0.1, 4, dim);
	mu_assert(rbm->first_layer != NULL, "didnot create initial layer");
	mu_assert(rbm->last_layer != NULL, "didnot set last layer");
	mu_assert(rbm->learning_rate == 0.1, "error rate not set");
	mu_assert(rbm->first_layer->dimension == 3, " initial layer dimension not set properly");
	mu_assert(rbm->first_layer->next->dimension == 4, "layer dimension not set properly");
	mu_assert(rbm->first_layer->next->next->dimension == 2, "layer dimension not set properly");
	mu_assert(rbm->first_layer->next->next->next == NULL, "layer");
	mu_assert(rbm->last_layer == rbm->first_layer->next->next, "layer");
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
	return NULL;
}

char * test_hidden_probablities_on_data()
{
	// SETUP
    // [m,n] matrix, [m,z] matrix, [1,z] vector, [1,n] vector, [n,z] matrix
    int m =100;
    int n = 300;
    int z = 50;
	unsigned int i, j, k;
	gsl_matrix * weights;
    gsl_rng * randgen;
    gsl_matrix * input;
    gsl_vector * hidden_bias;
    gsl_matrix * new_hidden;

    weights = gsl_matrix_alloc(n, z);
    randgen = gsl_rng_alloc (gsl_rng_taus); //fastest
    gsl_rng_set(randgen, (unsigned long int) time(NULL));
	for(j = 0; j < weights->size1; j++) {
		for(k = 0; k < weights->size2; k++) {
			gsl_matrix_set(weights, j, k, gsl_ran_gaussian(randgen,0.01));
       //     log_success("rando %g",gsl_ran_gaussian(randgen,0.01));
		}
	}
	input = gsl_matrix_alloc(m, n);
	for(j = 0; j < input->size1; j++) {
		for(k = 0; k < input->size2; k++) {
            //log_success("rando %d",gsl_rng_uniform_int(randgen,100));
			gsl_matrix_set(input, j, k, (double)gsl_rng_uniform_int(randgen,100));
		}
	}
	hidden_bias = gsl_vector_alloc(z);
    gsl_vector_set_all(hidden_bias,-1);
    // [m,z]
	new_hidden = get_batched_hidden_probablities_blas(input, hidden_bias, weights);
    // TESTING
	for(k = 0; k < input->size1; k++) { //each input [m,n]
        for(j = 0; j < hidden_bias->size; j++) { // bias [1,z] for each dimension of input
            double sum = 0;
            for(i = 0; i < weights->size1; i++) { 
                sum += gsl_matrix_get(weights,i,j) * gsl_matrix_get(input,k,i);
            }
            double partial = sum + gsl_vector_get(hidden_bias,j);
            double result = sigmoid(partial);
            mu_assert(float_check(result ,gsl_matrix_get(new_hidden,k,j)),"error in calc");
        }
    }
	log_success("Hidden probablity works");
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
	return NULL;
}
char * test_pretraining()
{
	gsl_matrix * test_visible = gsl_matrix_calloc(2, 4); // row major
	gsl_matrix * test_weights = gsl_matrix_calloc(4, 3);
	gsl_vector * test_hidden_bias = gsl_vector_calloc(3);
	gsl_vector * test_visible_bias = gsl_vector_calloc(4);
	gsl_vector_set_all(test_hidden_bias, 0.5);
	gsl_vector_set_all(test_visible_bias, 0.1);
	gsl_matrix_set_all(test_visible, 1);
	gsl_matrix_set_all(test_weights, 1);

	gsl_matrix * new_epoch = single_step_constrastive_convergence( test_visible, test_visible_bias, test_hidden_bias, test_weights, 0.1);
	mu_assert(new_epoch->size1 == test_weights->size1, "Wrong Dimensions");
	mu_assert(new_epoch->size2 == test_weights->size2, "Wrong Dimensions");
	log_warn("Small number of test cases");
	log_success("pretraining works");


	return NULL;
}



char *all_tests()
{
	mu_suite_start();
	mu_run_test(test_layer_creation);
	mu_run_test(test_hidden_probablities_blas);
	mu_run_test(test_visible_probablities_blas);
	mu_run_test(test_pretraining);
	mu_run_test(test_hidden_probablities_on_data);

	return NULL;
}

RUN_TESTS(all_tests);

