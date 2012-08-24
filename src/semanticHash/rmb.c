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

#define ARRAYSIZE(x)  (sizeof x / sizeof *x) 

/* RETURNS:
 *      A newly set up PretrainModel with the correct amount of layers, etc...
 *
 * NOTE: the weight matrix is stored column major.
 * TODO create function pointers for error_func, etc... */
PretrainModel * pretrain_create(
	double learning_rate, int input_dimension, int layer_dimension[])
{
	int i, num_of_layers;
	rbmLayer * p = NULL; // for pointer handshake
	PretrainModel * new_m;

    num_of_layers = ARRAYSIZE(layer_dimension);

    new_m = (PretrainModel *) malloc(sizeof(PretrainModel));
	new_m->learning_rate = learning_rate;
	new_m->first_layer = NULL;             // sentinals
	new_m->last_layer = NULL;


	for(i = 0; i < num_of_layers +1; i++) {
		rbmLayer * new_l =  (rbmLayer *) malloc( sizeof(rbmLayer));
		new_l->learning_rate = learning_rate;
		new_l->dimension = layer_dimension[i];
		new_l->hidden = gsl_vector_alloc(layer_dimension[i]);
		new_l->next = NULL;

		if(new_m->first_layer == NULL) {  // check to see if this matrix dimension is the initial input
			check_hard(i == 0, "should be first layer creation");
			new_l->weights = gsl_matrix_alloc(layer_dimension[i], input_dimension);

			new_l->prev = NULL;           // sentinal
			new_m->first_layer = new_l;   // make this the first layer

		} else {                          // have to be mindfull of any prev layer sizes
			check_hard(p != NULL, " no inital layer");
			new_l->weights = gsl_matrix_alloc(layer_dimension[i], p->dimension);

			p->next = new_l;
			new_l->prev = p;              //backlink for backpropagation, beause we run the rBm backwards too bro. lol
			new_m->last_layer = new_l;    //make this the last layer
		}

		p = new_l;                        // set handshake for next new layer
	}
	return new_m;                         // give back our newly created model
}

/*void pretrain_destroy(
	PretrainModel * model)
{
    model;
	//TODO
}*/

gsl_matrix * single_step_constrastive_convergence(
        gsl_matrix * input,
        gsl_vector * visible_bias, gsl_vector * hidden_bias,
        gsl_matrix * weights, double learning_rate)
{
    unsigned int i,j;
    // up
    gsl_matrix * hidden = get_batched_hidden_probablities_blas(input, hidden_bias, weights);
    for(i=0;i<hidden->size1;i++){
        for(j=0;j<hidden->size2;j++){
            double elem = gsl_matrix_get(hidden,i,j);
            if( 0.5 <= elem && elem < 1.0){
                gsl_matrix_set(hidden,i,j,1.0);
            } 
            else if(0 < elem && elem < 0.5)  {
                gsl_matrix_set(hidden,i,j,0);
            }else { // should never get here
                log_err("elem is out of bounds @: %g",elem);
            }
        }
    }
    // down
    gsl_matrix * visible_reconstructed = get_batched_visible_probablities_blas(input,hidden,visible_bias, weights);
    //up
    gsl_matrix * hidden_reconstructed = get_batched_hidden_probablities_blas(visible_reconstructed,hidden_bias,weights);

    // input = [a,b] hidden
    gsl_matrix_mul_elements(hidden, input);
    return hidden_reconstructed;
}

/* CALLED IN: single_step_constrastive_convergence.
 *
 * Paper: Ruslan Salakhutdinov and Geoffrey Hinton. Semantic hashing.
 * Equation Number: (2)
 *
 * PARAMS:
 *      [m,n] matrix, [1,z] vector , [n,z] matrix
 * RETURNS:
 *      [m,n] matrix.
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
        gsl_vector_view data = gsl_matrix_row(input,i);
        gsl_blas_dgemv(CblasTrans,1.0,weights,&data.vector, 1.0 ,temp);
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
 *      [m,n] matrix,
 * RETURNS:
 *      [m,n] matrix
 */
gsl_matrix * get_batched_visible_probablities_blas(
	gsl_matrix * input, gsl_matrix * hidden, gsl_vector * visible_bias, gsl_matrix * weights)
{
	unsigned int i, j;
	double numerator;
	double denominator;
	double multiplier;
	double sum;
	double result;
	double  * cached_results;
	gsl_matrix * visible;

	// create the cache for numerator
	cached_results = (double *)calloc(input->size1, sizeof(double));
	// create the matrix we eventually return
	visible = gsl_matrix_calloc(input->size1, input->size2);

	check_hard(hidden->size2 == weights->size2, "Mismatched hidden length with  weight matrix rows");
	check_hard(input->size2 == weights->size1, "Mismatched input rows with  weight matrix rows");
	check_hard(visible_bias->size == weights->size1, "Mismatched visible_bias length with  weight matrix rows");
	check_hard(hidden->size1 == input->size1, "Mismatched hidden  with  input in their rows");
	// Fill the cache for numerator/ denominator sum
	sum = 0;
	denominator = 0;
    gsl_vector * temp = gsl_vector_alloc(visible_bias->size);
	for(i = 0; i < input->size1; i++) { // for each input
        gsl_vector_memcpy(temp,visible_bias);
        gsl_vector_view hidden_vector = gsl_matrix_row(hidden,i); 
        gsl_blas_dgemv(CblasNoTrans,1.0,weights, &hidden_vector.vector, 1.0 ,temp);
        sum = exp_vector(temp);
        cached_results[i] = sum;
        denominator += sum;
    }

	for(i = 0; i < input->size1; i++) { // for each input
		gsl_vector_view  input_row = gsl_matrix_row(input, i);    //  get the single input
		numerator = cached_results[i]; // grab our previous computation
		multiplier = 0;
		for(j = 0; j < input->size2; j++) {
			multiplier += gsl_vector_get(&input_row.vector, j);
		}
        for(j = 0; j < input->size2; j++) {
           result = gsl_ran_poisson_pdf(gsl_vector_get(&input_row.vector, j), (numerator / denominator) * multiplier);
           gsl_matrix_set(visible,i,j, result);
        }
    }
	free(cached_results);
    gsl_vector_free(temp);
	return visible;
}

/********* BEGIN Helpers *********/

double exp_vector(gsl_vector * vec)
{
    unsigned int i = 0;
    double sum = 0;
    for(;i<vec->size;i++){
        sum += gsl_sf_exp(gsl_vector_get(vec,i));
    }
return sum;
}

double sigmoid(double x)
{
    //return 1 / (1 + gsl_sf_exp(-x)); dont use  unless underflow errors flag set when compiling
	return 1 / (1 + gsl_sf_exp(-x));
}
void sigmoid_vector(gsl_vector * vec)
{
    unsigned int i = 0;
    for(;i<vec->size;i++){
        gsl_vector_set(vec,i,sigmoid(gsl_vector_get(vec,i)));
    }
}
    
/* Equ (4) */
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

/* CALLED IN: single_step_constrastive_convergence.
 *
 * Paper: Ruslan Salakhutdinov and Geoffrey Hinton. Semantic hashing.
 * Equation Number: (2)
 *
 * PARAMS:
 *      [m,n] matrix, [1,z] vector , [z,n] matrix
 * RETURNS:
 *      [m,z] matrix.
 *
 * NOTE: the weight matrix needs to be in column major, with respect to the visible matrix.*/
/*gsl_matrix * get_batched_hidden_probablities(
	gsl_matrix * visible, gsl_vector * hidden_bias, gsl_matrix * weights)
{

	unsigned int i, j;
	double sum;
	double result;
	double final_result;
	gsl_matrix * hidden;
	hidden = gsl_matrix_calloc(visible->size1, hidden_bias->size);

	check_hard(weights->size1 == hidden_bias->size,
			   "Wrong Dimensions, weight_vec is: %zu, hidden_vec is %zu", weights->size1 , hidden_bias->size);
	check_hard(visible->size2 == weights->size2, "Visible and weight matrices don't match");

	for(i = 0; i < visible->size1; i++) { //for each visible entry
		gsl_vector_const_view  single_visible = gsl_matrix_const_row(visible, i);
		sum = 0;
		for(j = 0; j < hidden_bias->size; j++) { //for each dimension of bias
			gsl_vector_const_view  weight_vector = gsl_matrix_const_row(weights, j);
			gsl_blas_ddot(&weight_vector.vector, &single_visible.vector, &result);
			sum += result;
			final_result = sigmoid(gsl_vector_get(hidden_bias, j) + sum);
			gsl_matrix_set(hidden, i, j, final_result);
		}
	}
	return hidden;
}
*/
/* CALLED IN: single_step_constrastive_convergence.
 *
 * Paper: Ruslan Salakhutdinov and Geoffrey Hinton. Semantic hashing.
 * Equation Number: (1)
 *
 * PARAMS:
 *      [m,n] matrix, [1,z] vector , [1,z] vector , [z,n] matrix
 * RETURNS:
 *      [m,n] matrix
 *
 * NOTE: the weight matrix needs to be in column major, with respect to the input matrix.*/
/*gsl_matrix * get_batched_visible_probablities(
	gsl_matrix * input, gsl_matrix * hidden, gsl_vector * visible_bias, gsl_matrix * weights)
{
	unsigned int i, j;
	double numerator;
	double denominator;
	double multiplier;
	double sum;
	double result;
	double  * cached_results;
	gsl_matrix * visible;

	// create the cache for numerator
	cached_results = (double *)calloc(input->size1, sizeof(double));
	// create the matrix we eventually return
	visible = gsl_matrix_calloc(input->size1, visible_bias->size);

	// Fill the cache for numerator
	sum = 0;
	denominator = 0;
	//TODO use cblas level2?
	check_hard(hidden->size2 == weights->size1, "Mismatched hidden length with  weight matrix rows");
	check_hard(input->size2 == weights->size2, "Mismatched input rows with  weight matrix rows");
	check_hard(visible_bias->size == weights->size2, "Mismatched visible_bias length with  weight matrix rows");
	for(i = 0; i < input->size1; i++) { // for each input
		gsl_vector_view hidden_vector = gsl_matrix_row(hidden, i);
		for(j = 0; j < input->size2; j++) { // for each dimension of that single input
			gsl_vector_view weight_vector = gsl_matrix_column(weights, j);
			gsl_blas_ddot(&hidden_vector.vector, &weight_vector.vector , &result);
			sum += result;
		}
		result = gsl_sf_exp(gsl_vector_get(visible_bias, i) + sum);
		cached_results[i] = result;
		denominator += result;
	}


	for(i = 0; i < input->size1; i++) {
		gsl_vector_view  input_row = gsl_matrix_row(input, i);    // get the single data point to work with

		for(j = 0; j < input->size2; j++) {
			multiplier += gsl_vector_get(&input_row.vector, j);
		}
	}

	for(i = 0; i < input->size1; i++) { // for each input
		numerator = cached_results[i]; // grab our previous computation
		gsl_vector_view  input_row = gsl_matrix_row(input, i);    //  get the single input
		multiplier = 0;
		for(j = 0; j < input->size2; j++) {
			multiplier += gsl_vector_get(&input_row.vector, j);
		}
		for(j = 0; j < input->size2; j++) {
			gsl_matrix_set(visible, i, j,
						   gsl_ran_poisson_pdf(gsl_vector_get(&input_row.vector, j), numerator / denominator * multiplier));
		}
	}
	free(cached_results);
	return visible;
}
*/


