#include <assert.h>
#include <stdbool.h>
#include "minunit.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include "../src/semanticHash/dbg.h"
#include "../src/semanticHash/rmb.h"


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

	gsl_matrix * test_visible = gsl_matrix_alloc(2,4); // row major
	gsl_matrix * test_weights = gsl_matrix_alloc(4, 3);  //col major
	gsl_vector * test_hidden_bias = gsl_vector_calloc(3);

    gsl_matrix * new_hidden = get_batched_hidden_probablities_blas(test_visible, test_hidden_bias,test_weights);
    mu_assert(new_hidden->size1 == test_visible->size1, "Mismatched in to output row lengths");
    mu_assert(new_hidden->size2 == test_hidden_bias->size, "Mismatched column lengths");
    log_warn("Small number of test cases");
    log_success("hidden probablity works");
    return NULL;
}
char * test_visible_probablities_blas()
{

	gsl_matrix * test_visible = gsl_matrix_calloc(2,4); // row major
	gsl_matrix * test_weights = gsl_matrix_calloc(4, 3);  //col major
	gsl_vector * test_hidden_bias = gsl_vector_calloc(3);
	gsl_vector * test_visible_bias = gsl_vector_calloc(4);
    gsl_matrix_set_all(test_visible,1);
    gsl_matrix_set_all(test_weights,1);

    gsl_matrix * new_hidden = get_batched_hidden_probablities_blas(test_visible, test_hidden_bias,test_weights);
    gsl_matrix * new_visible = get_batched_visible_probablities_blas(test_visible, new_hidden, test_visible_bias,test_weights);
    mu_assert(new_visible->size1 == test_visible->size1, "Mismatched sizes");
    log_warn("Small number of test cases");
    log_success("Visible probablity works");
    return NULL;
}
/*char * test_hidden_probablities()
{

	int i, y;
	gsl_matrix * test_visible = gsl_matrix_alloc(2,4); // row major
	gsl_matrix * test_weights = gsl_matrix_alloc(3, 4);  //col major

	gsl_matrix_set_all(test_weights, 1.0);
	gsl_matrix_set_all(test_visible, 5.0);

	gsl_vector * test_hidden_bias = gsl_vector_calloc(3);
	gsl_vector * test_visible_bias = gsl_vector_calloc(4);

    gsl_matrix * new_hidden = get_batched_hidden_probablities(test_visible, test_hidden_bias, test_weights);
    mu_assert(new_hidden->size2 == 3, "different sizes for hidden and weight matrix size1");

    gsl_matrix * new_visible = get_batched_visible_probablities(test_visible,new_hidden,test_visible_bias,test_weights);
    mu_assert(new_visible->size1 == test_visible->size1, "new and old visible are Mismatched");
    mu_assert(new_visible->size2 == test_visible->size2, "new and old visible are Mismatched");
    log_success("hidden probablity model");
    return NULL;
}
*/
char * test_pretraining()
{
    log_warn("No tests for pretraining");
    return NULL;
}



char *all_tests()
{
	mu_suite_start();
    mu_run_test(test_hidden_probablities_blas);
    mu_run_test(test_visible_probablities_blas);
	//mu_run_test(test_layer_creation);
    //mu_run_test(test_hidden_probablities);
    //mu_run_test(test_pretraining);

	return NULL;
}

RUN_TESTS(all_tests);

