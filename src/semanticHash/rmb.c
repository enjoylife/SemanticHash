#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "dbg.h"
#include "rmb.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_randist.h>

#define ISABLELAYER 100
#define HIDDEN1 50

//TODO create function pointers for error_func, etc...
PretrainModel * pretrain_create(double learning_rate, int input_dimension, int layer_dimension[], int num_of_layers)
{
	int i;
	rbmLayer * p; // for pointer handshake
	PretrainModel * new_m = (PretrainModel *) malloc(sizeof(PretrainModel));
	new_m->learning_rate = learning_rate;
	new_m->first_layer = NULL;             // sentinals
	new_m->last_layer = NULL;

	for(i = 0; i < num_of_layers; i++) {
		rbmLayer * new_l =  (rbmLayer *) malloc( sizeof(rbmLayer));
		new_l->learning_rate = learning_rate;
		new_l->dimension = layer_dimension[i];
		new_l->hidden = gsl_vector_alloc(layer_dimension[i]);
		new_l->next = NULL;

		if(new_m->first_layer == NULL) {         // check to see if this matrix dimension is the initial input
			check_hard(i == 0, "should be first layer creation");
			new_l->weights = gsl_matrix_alloc(input_dimension, layer_dimension[i]);

			new_l->prev = NULL;           // sentinal
			new_m->first_layer = new_l;   // make this the first layer

		} else {                          // have to be mindfull of any prev layer sizes
			check_hard(p != NULL, " no inital layer");
			new_l->weights = gsl_matrix_alloc(p->dimension, layer_dimension[i]);

			p->next = new_l;
			new_l->prev = p;              //backlink for backpropagation, beause we run the rBm backwards too bro. lol
			new_m->last_layer = new_l;    //make this the last layer
		}

		p = new_l;                        // set handshake for next new layer
	}
	return new_m; // give back our newly created model
}
/* this is called for each training example in the first layer */
gsl_vector * first_layer_pretrain(
	double learning_rate,
	gsl_vector * visible,
	gsl_vector * visible_bias,
	gsl_vector * hidden_bias,
	gsl_matrix * weights)
{
	unsigned int i, y;
	gsl_vector * hidden;
	gsl_vector * visible_recon;
	gsl_vector * hidden_recon;
	double expect_data, expect_recon;

	gsl_matrix * m = gsl_matrix_alloc(visible->size, hidden->size);
	gsl_matrix_set_all(m, 1.0);

	hidden = gsl_vector_alloc(hidden_bias->size);
	visible_recon = gsl_vector_alloc(visible->size);
	hidden_recon = gsl_vector_alloc(hidden_bias->size);

	/* Upward */
	for(y = 0; y < hidden->size; y++) {
		// create the hidden vector
		gsl_vector_set(hidden, y, hidden_probablity_model(y, hidden_bias, visible, weights));
	}

	/* Downward */
	for(i = 0; i < visible_recon->size; i++) {
		// create the recontructed visible vectors
		gsl_vector_set(visible_recon, i, visible_probablity_model(i, visible, visible_bias, hidden, weights));
	}
	/* Upward */
	for(y = 0; y < hidden->size; y++) {
		// update the hidden vectors again this time with recontructed prob
		gsl_vector_set(hidden, y, hidden_probablity_model(y, hidden_bias, visible_recon, weights));
	}

	for(i = 0; i < visible->size; i++) {
		for(y = 0; y < hidden->size; y++) {
			expect_data = gsl_vector_get(visible, i) * gsl_vector_get(hidden, y);
			expect_recon = gsl_vector_get(visible_recon, i) * gsl_vector_get(hidden_recon, y);
			gsl_matrix_set(m, i, y, expect_data - expect_recon);
		}
	}
	// used  as the next layers inital visible units
	return hidden_recon;
}

/* Computes  P(v_i |h)  where v_i = visible data point, and h = hidden vector
* Equ (1) */
double visible_probablity_model(
	int visible_sample_index,
	gsl_vector * visible,
	gsl_vector * visible_bias,
	gsl_vector * hidden,
	gsl_matrix * weights)
{
	gsl_vector * intermediate;
	double numerator, denominator;
	double multiplier;
	double sum;
	unsigned int i;

	assert(visible->size == visible_bias->size);
	intermediate = gsl_vector_alloc(hidden->size);
	gsl_blas_dgemv(CblasNoTrans, 1.0, weights, hidden, 1.0, intermediate);

	//TODO Move this outside so we don't have to compute again per visible vector element
	sum = 0;
	for(i = 0; i < intermediate->size; i++) {
		sum += gsl_vector_get(intermediate, i);
	}

	numerator = gsl_sf_exp( gsl_vector_get(visible_bias, visible_sample_index) + sum);

	//TODO Move this outside so we don't have to compute again per visible vector element
	denominator = 0;
	for(i = 0; i < visible_bias->size; i++) {
		denominator += gsl_sf_exp(gsl_vector_get(visible_bias, i) + sum);
	}

	multiplier = 0;
	for(i = 0; i < visible->size; i++) {
		multiplier += gsl_vector_get(visible, i);
	}

	gsl_vector_free(intermediate);
	return gsl_ran_poisson_pdf(gsl_vector_get(visible, visible_sample_index), numerator / denominator * multiplier);
}


/* Equ (2) */
double hidden_probablity_model(
	int hidden_sample_index,
	gsl_vector * hidden_bias,
	gsl_vector * visible,
	gsl_matrix * weights)
{

	int i;
	double sum;
	gsl_vector * intermediate;
	intermediate = gsl_vector_calloc(hidden_bias->size);

	//log_info("visible vector length: %d", visible->size);
	//log_info("hidden_bias length: %d", hidden_bias->size);
	//log_info("intermediate vector length: %d", intermediate->size);
	//log_info("matrix dimensions are: %d, %d", weights->size1, weights->size2);

	gsl_blas_dgemv(CblasTrans, 1.0, weights, visible, 1.0, intermediate);
	sum = 0;
	for(i = 0; i < intermediate->size; i++) {
		sum += gsl_vector_get(intermediate, i);
	}

	gsl_vector_free(intermediate);
	return 	sigmoid(gsl_sf_exp(gsl_vector_get(hidden_bias, hidden_sample_index) + sum));
}

double sigmoid(double x)
{
	// dont use gsl exp, unless you turn off underflow errors when compiling
	//return 1 / (1 + gsl_sf_exp(-x));
	log_info("sigmoid out is %g", 1 / (1 + exp(-x)));
	return 1 / (1 + exp(-x));
}


/* Equ (4) */
double energy(gsl_vector * visible, gsl_vector * visible_bias,
			  gsl_vector * hidden, gsl_vector * hidden_bias,
			  gsl_matrix * weights)
{
	double visible_sum = 0;
	double sum1 = 0;
	double sum2 = 0;
	double sum3 = 0;
	double sum4 = 0;
	double sum5 = 0;
	double sum2_log = 0;
	double v, h;
	unsigned int i, j ;

	for(i = 0; i < visible->size; i++) {
		visible_sum += gsl_vector_get(visible, i);
	}
	for(i = 0; i < visible->size; i++) {
		sum2_log += gsl_sf_exp(gsl_vector_get(visible_bias, i) +
							   gsl_blas_dtrmv(CblasUpper, CblasNoTrans, CblasNonUnit, weights, hidden));
	}
	for(i = 0; i < visible->size; i++) {
		v = gsl_vector_get(visible, i);
		sum1 -=  v * gsl_vector_get(visible_bias, i);
		sum2 += v * gsl_sf_log(sum2_log / visible_sum);
		// faster for values larger than 170. . . so it says in gnu gsl docs?
		sum3 += gsl_sf_lngamma(v + 1);
		for(j = 0; j < hidden->size; j++) {
			h = gsl_vector_get(hidden, j);
			sum4 -= h * gsl_vector_get(hidden_bias, j);
			sum5 -= v * h * gsl_matrix_get(weights, i, j);
		}
	}
	return sum1 + sum2 + sum3 + sum4 + sum5;
}

/* Equ (6) */
double deep_visible_update_rule(
	int sample_index,
	gsl_vector * vec, gsl_vector * bias,
	gsl_matrix * weights)
{
	return 	sigmoid(
				gsl_sf_exp( gsl_vector_get(bias, sample_index) +
							gsl_blas_dtrmv(CblasUpper, CblasTrans, CblasNonUnit, weights, vec)));
}
/* Equ (7) */
double deep_hidden_update_rule(
	int hidden_sample_index,
	gsl_vector * hidden, gsl_vector * hidden_bias,
	gsl_matrix * weights)
{
	return sigmoid(
			   gsl_sf_exp( gsl_vector_get(hidden_bias, hidden_sample_index) +
						   gsl_blas_dtrmv(CblasUpper, CblasNoTrans, CblasNonUnit, weights, hidden)));
}

