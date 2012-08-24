#ifndef semanticHash_List_h
#define semanticHash_List_h
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

typedef struct rbmLayer {

	double learning_rate;
	int dimension;
	gsl_matrix * weights; // dimensions  = [prev->dimension, dimension]
	gsl_vector * hidden;
	struct rbmLayer * next;
	struct rbmLayer * prev;
} rbmLayer;

typedef struct PretrainModel {

	double learning_rate;
	rbmLayer * first_layer;
	rbmLayer * last_layer;
}PretrainModel;

PretrainModel * pretrain_create(double learning_rate, int input_dimension, int layer_dimension[]);

/*gsl_matrix * get_batched_hidden_probablities(
        gsl_matrix * visible, gsl_vector * hidden_bias, gsl_matrix * weights);
gsl_matrix * get_batched_visible_probablities(
        gsl_matrix * input, gsl_matrix * hidden, gsl_vector * visible_bias, gsl_matrix * weights);
        */
gsl_matrix * get_batched_hidden_probablities_blas(
	gsl_matrix * input, gsl_vector * hidden_bias, gsl_matrix * weights);
gsl_matrix * get_batched_visible_probablities_blas(
	gsl_matrix * input, gsl_matrix * hidden, gsl_vector * visible_bias, gsl_matrix * weights);

double exp_vector(gsl_vector * vec);
double sigmoid(double x);
void sigmoid_vector(gsl_vector * vec);

double visible_probablity_model(
	int visible_sample_index,
	gsl_vector * visible, gsl_vector * visible_bias,
	gsl_vector * hidden, gsl_matrix * weights);

double hidden_probablity_model(
	int hidden_sample_index,
	gsl_vector * hidden_bias,
	gsl_vector * visible,
	gsl_matrix * weights);

double energy(gsl_vector * visible, gsl_vector * visible_bias,
			  gsl_vector * hidden, gsl_vector * hidden_bias,
			  gsl_matrix * weights);

#endif
