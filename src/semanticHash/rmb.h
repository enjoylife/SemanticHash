#ifndef semanticHash_List_h
#define semanticHash_List_h
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#define FIRSTLAYER 0
#define GENERAL 1

/* one of these struct's per layer, independent of layer type, size, etc */
typedef struct rbmLayer {
	double learning_rate;
	int dimension;
	gsl_matrix * weights; // dimensions  = [prev->dimension, dimension]
	gsl_vector * hidden_bias;
    gsl_vector * visible_bias;
	struct rbmLayer * next;
	struct rbmLayer * prev;
} rbmLayer;

/* the main struct of the pretrain phase, basiclly is just a linked list of rbmLayer's */
typedef struct PretrainModel {
	double learning_rate;
	rbmLayer * first_layer;
	rbmLayer * last_layer;
} PretrainModel;

/* This should fullfill the pretrain portion of the semantic hashing algorithm. */
void pretrain( PretrainModel * test, FILE * input_file, int batch_size, int single_layer_epoch);

/* should be followed by a pretrain_destroy when done */
PretrainModel * pretrain_create(double learning_rate, int layer_dimension[]);

void pretrain_destroy(PretrainModel * model);

void single_step_constrastive_convergence(
	gsl_matrix * input,
	gsl_vector * visible_bias,
    gsl_vector * hidden_bias,
	gsl_matrix * weights,
    double learning_rate, int layer_type);

gsl_matrix * get_batched_general_probablities_blas(
	gsl_matrix * input, gsl_vector * visible_bias, gsl_matrix * weights);

gsl_matrix * get_batched_hidden_probablities_blas(
	gsl_matrix * input, gsl_vector * hidden_bias, gsl_matrix * weights);

gsl_matrix * get_batched_visible_probablities_blas(
	gsl_matrix * input, gsl_matrix * hidden, gsl_vector * visible_bias, gsl_matrix * weights);

int  read_in_batch(gsl_matrix * input_matrix, FILE * input_file, int batch_size, int input_dimension);

void exp_vector(gsl_vector * vec);

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
