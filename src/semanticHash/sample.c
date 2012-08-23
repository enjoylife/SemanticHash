/* insert sort */

#include <stdlib.h>
#include <stdio.h>

//#define SWAP(x, y) do { typeof(x) temp##x##y = x; x = y; y = temp##x##y; } while (0)

typedef int T;          /* type of item to be sorted */
typedef int tblIndex;   /* type of subscript */

#define compGT(a,b) (a > b)

void insertSort(T *a, tblIndex lb, tblIndex ub) {
    T t;
    tblIndex i, j;

   /**************************
    *  sort array a[lb..ub]  *
    **************************/
    for (i = lb + 1; i <= ub; i++) {
        t = a[i];

        /* Shift elements down until */
        /* insertion point found.    */
        for (j = i-1; j >= lb && compGT(a[j], t); j--)
            a[j+1] = a[j];

        /* insert */
        a[j+1] = t;
    }
}

void fill(T *a, tblIndex lb, tblIndex ub) {
    tblIndex i;
    srand(1);
    for (i = lb; i <= ub; i++) a[i] = rand();
}

/* negative if the first argument is “less” than the second,
 * zero if they are “equal”,
 * and positive if the first argument is “greater”. */
int compare_doubles (void *a,void *b)
{
	const double *da = (const double *) a;
	const double *db = (const double *) b;

	return (*da > *db) - (*da < *db);
}
int main(int argc, char *argv[]) {
    tblIndex maxnum, lb, ub;
    T *a;
    int i= 0;

    /* command-line:
     *
     *   ins maxnum
     *
     *   ins 2000
     *       sorts 2000 records
     *
     */

    maxnum = atoi(argv[1]);
    lb = 0; ub = maxnum - 1;
    if ((a = malloc(maxnum * sizeof(T))) == 0) {
        fprintf (stderr, "insufficient memory (a)\n");
        exit(1);
    }

    fill(a, lb, ub);
   
    insertSort(a, lb, ub);
    /*for(i=0;i<maxnum; i++){
    printf("%d \n",a[i]);
    };*/

    return 0;
}
