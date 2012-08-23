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
	PretrainModel * rbm = pretrain_create(0.1, 4, dim, 3);
	mu_assert(rbm->first_layer != NULL, "didnot create initial layer");
	mu_assert(rbm->last_layer != NULL, "didnot set last layer");
	log_warn("rate: %f", rbm->learning_rate);
	mu_assert(rbm->learning_rate == 0.1, "error rate not set");
	mu_assert(rbm->first_layer->dimension == 3, " initial layer dimension not set properly");
	mu_assert(rbm->first_layer->next->dimension == 4, "layer dimension not set properly");
	mu_assert(rbm->first_layer->next->next->dimension == 2, "layer dimension not set properly");
	mu_assert(rbm->first_layer->next->next->next == NULL, "layer");
	mu_assert(rbm->last_layer == rbm->first_layer->next->next, "layer");
	return NULL;
}

char * test_pretraining()
{
    return NULL;
}

char * constrained_poisson_model()
{
	int i, y;
	gsl_vector * test_visible = gsl_vector_alloc(4);
	gsl_vector * test_visible_bias = gsl_vector_calloc(4);
	gsl_vector * test_hidden_bias = gsl_vector_calloc(3);
	gsl_matrix * test_weights = gsl_matrix_alloc(4, 3);
	gsl_matrix_set_all(test_weights, 5.0);
	for(i = 0; i < 4; i++) {
		gsl_vector_set(test_visible, i, 7);
	}
	double out = hidden_probablity_model( 0, test_hidden_bias, test_visible, test_weights);
	mu_assert(test_visible != NULL, " THISISDIFFERENT");
	return NULL;
}

char *all_tests()
{
	mu_suite_start();
	mu_run_test(constrained_poisson_model);
	mu_run_test(test_layer_creation);
    mu_run_test(test_pretraining);

	return NULL;
}

RUN_TESTS(all_tests);

