/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

/*
 * Sort array using blocking send/recv between 2 ranks.
 *
 * The master process prepares the data and sends the latter half
 * of the array to the other rank. Each rank sorts it half. The
 * master then merges the sorted halves together. The two ranks
 * communicate using blocking send/recv.
 */

#define NUM_ELEMENTS 50

static int compare_int(const void *a, const void *b)
{
    return (*(int *) a - *(int *) b);
}

/* Merge sorted arrays a[] and b[] into a[].
 * Length of a[] must be sum of lengths of a[] and b[] */
static void merge(int *a, int numel_a, int *b, int numel_b)
{
    int *sorted = (int *) malloc((numel_a + numel_b) * sizeof *a);
    int i, a_i = 0, b_i = 0;
    /* merge a[] and b[] into sorted[] */
    for (i = 0; i < (numel_a + numel_b); i++) {
        if (a_i < numel_a && b_i < numel_b) {
            if (a[a_i] < b[b_i]) {
                sorted[i] = a[a_i];
                a_i++;
            } else {
                sorted[i] = b[b_i];
                b_i++;
            }
        } else {
            if (a_i < numel_a) {
                sorted[i] = a[a_i];
                a_i++;
            } else {
                sorted[i] = b[b_i];
                b_i++;
            }
        }
    }
    /* copy sorted[] into a[] */
    memcpy(a, sorted, (numel_a + numel_b) * sizeof *sorted);
    free(sorted);
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) printf("Please run with exactly 2 ranks\n");
        MPI_Finalize();
        return 0;
    }

    // For MPI_Scatter / MPI_Gather, each rank must receive/send the same count.
    // NUM_ELEMENTS must be divisible by 2.
    if (NUM_ELEMENTS % 2 != 0) {
        if (rank == 0) printf("NUM_ELEMENTS must be divisible by 2 for MPI_Scatter/MPI_Gather\n");
        MPI_Finalize();
        return 0;
    }

    int half = NUM_ELEMENTS / 2;

    int data[NUM_ELEMENTS];          // only meaningful on rank 0 before scatter, and after gather
    int local[NUM_ELEMENTS / 2];     // each rank receives 25 elements here

    srand(0);

    if (rank == 0) {
        printf("Unsorted:\t");
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            data[i] = rand() % NUM_ELEMENTS;
            printf("%d ", data[i]);
        }
        printf("\n");
    }

    // Scatter: rank 0 sends first 25 ints to rank 0, second 25 ints to rank 1
    MPI_Scatter(
        data,            // send buffer (rank 0 only)
        half, MPI_INT,   // send count/type per rank
        local,           // receive buffer (all ranks)
        half, MPI_INT,   // receive count/type
        0,               // root
        MPI_COMM_WORLD
    );

    // Each rank sorts its local chunk
    qsort(local, half, sizeof(int), compare_int);

    // Gather: collect sorted chunks back into data on rank 0
    MPI_Gather(
        local,           // send buffer (each rank)
        half, MPI_INT,   // send count/type
        data,            // recv buffer (rank 0)
        half, MPI_INT,   // recv count/type per rank
        0,               // root
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        // Now data[0..24] and data[25..49] are each sorted, but the whole array isn't.
        // Merge the two sorted halves into a fully sorted array.
        merge(data, half, &data[half], half);

        printf("Sorted:\t\t");
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            printf("%d ", data[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}