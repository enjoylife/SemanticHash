#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <ctype.h>
#include "dbg.h"
#include "rmb.h"
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_sf_exp.h>
#include <gsl/gsl_sf_log.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

#define ARRAYSIZE(x)  (sizeof x / sizeof *x)


/* We need to go through each layer single_layer_epoch number of times, updating the weights 
 * after a batch_size of input. We do this for each layer in the PretrainModel.
 * Any error we fail hard and bail the rest of training.
 * TODO: Incorperate training config options for custom learning stuff. */
void pretrain( PretrainModel * test, FILE * input_file, int batch_size, int single_layer_epoch)
{
	int k; // epoch counter
    int i;
    int j, size, data, layer_type;
    int batch_check;
	gsl_matrix * input;
	char ** input_line;

	check_hard(input_file, "File pointer not valid");
	rbmLayer * layer = test->first_layer;
	check_hard(layer, "File pointer not valid");

	while(layer) { // we have more layers to train
		input = gsl_matrix_alloc(batch_size, layer->dimension);  //dimension for each layer

		if(layer == test->first_layer) { // need to read in from file
			for(k = 0; k < single_layer_epoch; k++) {
				rewind(input_file); // we need all input
                batch_check = read_in_batch(input, input_file, batch_size, layer->dimension);
				if(batch_check != batch_size) { // we need a view of the matrices for only the rows of data read in
					gsl_matrix_view partial_batch_input = gsl_matrix_submatrix(input, 0, 0, batch_check, j);
					gsl_matrix_view partial_batch_weights = gsl_matrix_submatrix(layer->weights, 0, 0, batch_check, j);
					single_step_constrastive_convergence(
						&partial_batch_input.matrix, layer->visible_bias, layer->hidden_bias,
						&partial_batch_weights.matrix, layer->learning_rate, FIRSTLAYER);

				} else {
					single_step_constrastive_convergence(
						input, layer->visible_bias, layer->hidden_bias, layer->weights, layer->learning_rate, FIRSTLAYER);
				}

			} // all epochs complete
		} // finished with FIRSTLAYER
		// TODO The rest of the other layer learning
		else {
			layer_type = GENERAL;
		}
		gsl_matrix_free(input);
		layer = layer->next;
	}
}


/* Has to deal with having non divisible number of input lines with respect to batch_size
 * so it returns the actual size read in, which may be different than the batch_size. */
int  read_in_batch(gsl_matrix * input_matrix, FILE * input_file, int batch_size, int input_dimension){
    int i, j, size;
    double value;
    char *delim = " "; // input separated by spaces
    char *token = NULL;
    char *unconverted;
    size_t nbytes = 100;
    char *input_line;
    input_line = (char *) malloc(nbytes);

    check_hard(input_file,"Invalid file");
    check_hard(input_matrix->size1 == batch_size,"wrong row dimension");
    check_hard(input_matrix->size2 == input_dimension,"wrong column dimension");

    for(i = 0; i < batch_size; i++) { // we may read in less than batch_size lines
        size = getline(&input_line, &nbytes, input_file);
        check_hard(input_line,"line ptr not alloced");
        check_hard(*input_line,"line not alloced");
        //log_info("string is %s",input_line);
        if(size == -1) { // end of file
            break;
        }
        for (j = 0, token = strtok(input_line, delim); token != NULL; token = strtok(NULL, delim),j++)
        {
            check_hard(j <= input_dimension,"out of bounds for column length");
            value = strtod(token, &unconverted);
            //check_hard(!isspace(*unconverted) && *unconverted != 0,"Input string contains invalid char");
            gsl_matrix_set(input_matrix,i,j,value);
        }
    } 
    free(input_line);
    return i; // total number of rows read in
}
/* RETURNS:
 *      A newly set up PretrainModel with the correct amount of layers, etc...
 *
 * NOTE: the weight matrix is stored column major.
 * TODO create function pointers for custom stuff; error_func, parameter printing, etc... */
PretrainModel * pretrain_create(
	double learning_rate, int layer_dimension[])
{
	int i, num_of_layers;
	rbmLayer * p = NULL;                  // for pointer handshake
	PretrainModel * new_m;

	num_of_layers = ARRAYSIZE(layer_dimension);
	check_hard(num_of_layers >= 2, "Need at least an input layer and 1 hidden");

	new_m = (PretrainModel *) malloc(sizeof(PretrainModel));
	new_m->learning_rate = learning_rate;
	new_m->first_layer = NULL;             // sentinals
	new_m->last_layer = NULL;


	for(i = 0; i < num_of_layers + 1; i++) { // zero indexed arrays
		rbmLayer * new_l =  (rbmLayer *) malloc( sizeof(rbmLayer));
		new_l->learning_rate = learning_rate;
		new_l->hidden_bias = gsl_vector_calloc(layer_dimension[i]);
		new_l->dimension = layer_dimension[i];
		new_l->next = NULL;

		if(new_m->first_layer == NULL) {
			check_hard(i == 0, "should be first layer creation");
			new_l->weights = gsl_matrix_calloc(layer_dimension[i], layer_dimension[i + 1]);
			new_l->visible_bias = gsl_vector_calloc(layer_dimension[i]);
			new_l->prev = NULL;           // sentinal
			new_m->first_layer = new_l;   // make this the first layer

		} else {                          // have to be mindfull of any prev layer sizes
			check_hard(p != NULL, " no inital layer");
			new_l->weights = gsl_matrix_alloc(p->dimension, layer_dimension[i]);
			new_l->visible_bias = gsl_vector_calloc(p->dimension);

			p->next = new_l;
			new_l->prev = p;              //backlink for backpropagation
			new_m->last_layer = new_l;    //make this the last layer
		}

		p = new_l;                        // set handshake for next new layer
	}
	return new_m;                         // give back our newly created model
}

void single_step_constrastive_convergence(
	gsl_matrix * input,
	gsl_vector * visible_bias, gsl_vector * hidden_bias,
	gsl_matrix * weights, double learning_rate, int layer_type)
{
	unsigned int i, j;
	gsl_rng * randgen;
	gsl_matrix * visible_reconstructed;
	randgen = gsl_rng_alloc (gsl_rng_taus); //fastest
	gsl_rng_set(randgen, (unsigned long int) 10);//time(NULL));
	// up
	gsl_matrix * hidden = get_batched_hidden_probablities_blas(input, hidden_bias, weights);
	// TODO Compare to a zero mean gaussian like in the matlab code
	for(i = 0; i < hidden->size1; i++) {
		for(j = 0; j < hidden->size2; j++) {
			double elem = gsl_matrix_get(hidden, i, j);
			double rando = gsl_ran_gaussian(randgen, 0.01);
			( rando < elem ) ? gsl_matrix_set(hidden, i, j, 1.0) : gsl_matrix_set(hidden, i, j, 0);
		}
	}
	if(layer_type == FIRSTLAYER) {
		// down
		visible_reconstructed = get_batched_visible_probablities_blas(input, hidden, visible_bias, weights);
	} else if(layer_type == GENERAL) {
		//up
		//visible_reconstructed = get_batched_general_probablities_blas(
	}

	gsl_matrix * hidden_reconstructed = get_batched_hidden_probablities_blas(visible_reconstructed, hidden_bias, weights);

	gsl_matrix * temp_data = gsl_matrix_calloc(weights->size1, weights->size2);
	gsl_matrix * temp_recon = gsl_matrix_calloc(weights->size1, weights->size2);
	// visible [m,n] matrix, hidden [m,z] matrix,
	// visible [1,z] hidden  [1,n]
	// weights [n,z] matrix

	check_hard(input->size1 == hidden->size1, "mismatched dimensions");
	check_hard(visible_reconstructed->size1 == hidden_reconstructed->size1, "mismatched dimensions");

	//log_info("input dimensions [%ld,%ld]", input->size1, input->size2);
	//log_info("hidden dimensions [%ld,%ld]", hidden->size1, hidden->size2);
	// updates

	//multiply
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, input, hidden, 1.0, temp_data);
	gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, visible_reconstructed, hidden_reconstructed, 1.0, temp_recon);
	// subtract
	gsl_matrix_sub(temp_data, temp_recon);
	//scale
	gsl_matrix_scale(temp_data, 1 / input->size1); //scale by batch size
	gsl_matrix_scale(temp_data, learning_rate);
	// finally the actual update
	gsl_matrix_add(weights, temp_data);

	// update hidden_bias
	gsl_matrix_sub(hidden, hidden_reconstructed);
	for(i = 0; i < hidden->size2; i++) {
		double sum = 0;
		for(j = 0; j < hidden->size1; j++) {
			//flatten the matrix
			sum += gsl_matrix_get(hidden, j, i);
		}
		gsl_vector_set(hidden_bias, i, sum);
	}
	gsl_vector_scale(hidden_bias, 1 / input->size1);
	gsl_vector_scale(hidden_bias, learning_rate);

	// upate visible_bias
	gsl_matrix_sub(input, visible_reconstructed);
	for(i = 0; i < input->size2; i++) {
		double sum = 0;
		for(j = 0; j < input->size1; j++) {
			//flatten the matrix
			sum += gsl_matrix_get(input, j, i);
		}
		gsl_vector_set(visible_bias, i, sum);
	}
	gsl_vector_scale(visible_bias, 1 / input->size1);
	gsl_vector_scale(visible_bias, learning_rate);

	gsl_matrix_free(temp_recon);
	gsl_matrix_free(temp_data);
	gsl_rng_free(randgen);
}

/* Rember to destroy the matrix that is returned by this func */
gsl_matrix * get_batched_general_probablities_blas(
	gsl_matrix * input, gsl_vector * visible_bias, gsl_matrix * weights)
{
	check_hard(visible_bias->size == weights->size1, "Mismatched bias to weights");
	check_hard(input->size2 == weights->size1, "Mismatched input to weights");

	unsigned int i;
	gsl_vector * temp = gsl_vector_alloc(visible_bias->size);
	gsl_matrix * hidden = gsl_matrix_alloc(input->size1, visible_bias->size);
	for(i = 0; i < input->size1; i++) { //for each visible entry
		gsl_vector_memcpy(temp, visible_bias); // temp gets overwritten each call
		gsl_vector_view data = gsl_matrix_row(input, i);
		gsl_blas_dgemv(CblasNoTrans, 1.0, weights, &data.vector, 1.0 , temp);
		sigmoid_vector(temp);
		gsl_matrix_set_row(hidden, i, temp);
	}
	gsl_vector_free(temp);
	return hidden;
}

/* CALLED IN: single_step_constrastive_convergence.
 *
 * Paper: Ruslan Salakhutdinov and Geoffrey Hinton. Semantic hashing.
 * Equation Number: (2)
 *
 * PARAMS:
 * RETURNS:
 * NOTE: remember to destroy the matrix that is returned by this func.
 */
gsl_matrix * get_batched_hidden_probablities_blas(
	gsl_matrix * input, gsl_vector * hidden_bias, gsl_matrix * weights)
{
	check_hard(hidden_bias->size == weights->size2, "Mismatched bias to weights");
	check_hard(input->size2 == weights->size1, "Mismatched input to weights");

	unsigned int i;
	gsl_vector * temp = gsl_vector_alloc(hidden_bias->size);
	gsl_matrix * hidden = gsl_matrix_alloc(input->size1, hidden_bias->size);
	for(i = 0; i < input->size1; i++) { //for each visible entry
		gsl_vector_memcpy(temp, hidden_bias); // temp gets overwritten each call
		gsl_vector_view data = gsl_matrix_row(input, i);
		gsl_blas_dgemv(CblasTrans, 1.0, weights, &data.vector, 1.0 , temp);
		sigmoid_vector(temp);
		gsl_matrix_set_row(hidden, i, temp);
	}
	gsl_vector_free(temp);
	return hidden;
}

/* CALLED IN: single_step_constrastive_convergence.
 *
 * Paper: Ruslan Salakhutdinov and Geoffrey Hinton. Semantic hashing.
 * Equation Number: (1)
 *
 * PARAMS:
 *      [m,n] matrix, [m,z] matrix, [1,z] vector, [1,n] vector, [n,z] matrix
 * RETURNS:
 *      [m,n] matrix
 */
gsl_matrix * get_batched_visible_probablities_blas(
	gsl_matrix * input, gsl_matrix * hidden, gsl_vector * visible_bias, gsl_matrix * weights)
{
	unsigned int i, j, k, v;
	double numerator;
	double multiplier;
	double sum;
	double result;
	gsl_matrix * cached_results;
	gsl_matrix * visible;

	// create the cache for numerator
	cached_results = gsl_matrix_calloc(input->size1, input->size2);
	// create the matrix we eventually return
	visible = gsl_matrix_calloc(input->size1, input->size2);
	// create denominator for holding denominators for each input
	double  * denominator = (double*) calloc(input->size1,  sizeof(double));

	check_hard(hidden->size2 == weights->size2, "Mismatched hidden length with  weight matrix rows");
	check_hard(input->size2 == weights->size1, "Mismatched input rows with  weight matrix rows");
	check_hard(visible_bias->size == weights->size1, "Mismatched visible_bias length with  weight matrix rows");
	check_hard(visible_bias->size == input->size2, "Mismatched visible_bias length with  weight matrix rows");
	check_hard(hidden->size1 == input->size1, "Mismatched hidden  with  input in their rows");
	// Fill the cache for numerator/ denominator sum
	gsl_vector * temp = gsl_vector_alloc(visible_bias->size);
	// create the denominators and cache for total batch
	for(i = 0; i < input->size1; i++) { // for each input
		gsl_vector_memcpy(temp, visible_bias); // give it the correct visible bias
		gsl_vector_view hidden_vector = gsl_matrix_row(hidden, i); //  give it the correct hidden_bias
		gsl_blas_dgemv(CblasNoTrans, 1.0, weights, &hidden_vector.vector, 1.0 , temp);

		exp_vector(temp);
		sum = 0;
		for(j = 0; j < temp->size; j++) {
			sum += gsl_vector_get(temp, j);
		}
		gsl_matrix_set_row(cached_results, i, temp);
		denominator[i] = sum;
	}

	for(i = 0; i < input->size1; i++) { // for each input
		gsl_vector_view  input_row = gsl_matrix_row(input, i);    //  get the single input
		multiplier = 0;
		// multiplier to be used in all columns of this input
		for(j = 0; j < input->size2; j++) {
			multiplier += gsl_vector_get(&input_row.vector, j);
		}
		for(j = 0; j < input->size2; j++) {
			result = gsl_ran_poisson_pdf(gsl_vector_get(&input_row.vector, j), // single column
										 (gsl_matrix_get(cached_results, i, j) / denominator[i]) * multiplier);
			gsl_matrix_set(visible, i, j, result);
			// log_warn("real num: %g", gsl_matrix_get(cached_results,i,j));
			// log_warn("real denom: %g", denominator[i]);
			// log_warn("real multi: %g", multiplier);

		}
	}
	free(denominator);
	gsl_matrix_free(cached_results);
	gsl_vector_free(temp);
	return visible;
}

/********* BEGIN Helpers *********/

void  exp_vector(gsl_vector * vec)
{
	unsigned int i = 0;
	for(; i < vec->size; i++) {
		gsl_vector_set(vec, i, gsl_sf_exp(gsl_vector_get(vec, i)));
	}
}

void sigmoid_vector(gsl_vector * vec)
{
	unsigned int i = 0;
	for(; i < vec->size; i++) {
		gsl_vector_set(vec, i, sigmoid(gsl_vector_get(vec, i)));
	}
}
double sigmoid(double x)
{
	//return 1 / (1 + gsl_sf_exp(-x)); dont use  unless underflow errors flag set when compiling
	return 1 / (1 + gsl_sf_exp(-x));
}

/* Equ (4),
 * Does not work.
 * TODO: Implement */
double energy(
	gsl_vector * visible, gsl_vector * visible_bias,
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

/********* END Helpers *********/
