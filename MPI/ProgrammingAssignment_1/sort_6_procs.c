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
 * Sort array using blocking send/recv between 6 ranks.
 *
 * The master process prepares the data and sends the other parts 
 * of the array to other ranks. Each rank sorts its part. The 
 * master then merges the sorted parts together. All ranks
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

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 6) {
        if (rank == 0) printf("Please run with exactly 6 ranks\n");
        MPI_Finalize();
        return 0;
    }

    // count[] is used to store number of elements for each rank 
    // displs[] is used to starting index for each rank in the original array
    int counts[6], displs[6];
    int base = NUM_ELEMENTS / size;   // 50/6 = 8
    int rem  = NUM_ELEMENTS % size;   // 50%6 = 2 (these are split to rank0 and rank 1)

    int off = 0;
    for (int r = 0; r < size; r++) {
        counts[r] = base + (r < rem ? 1 : 0);  //rank 0 and 1 gets 9 elements, others get 8 elements
        displs[r] = off;
        off += counts[r];
    }

    int local_n = counts[rank];
    int *local = (int *)malloc(local_n * sizeof(int));

    // Rank 0 owns full data
    int data[NUM_ELEMENTS];

    if (rank == 0) {
        srand(0);
        printf("Unsorted:\t");
        for (int i = 0; i < NUM_ELEMENTS; i++) {
            data[i] = rand() % NUM_ELEMENTS;
            printf("%d ", data[i]);
        }
        printf("\n");

        // Copy rank0's own chunk into local
        memcpy(local, &data[displs[0]], local_n * sizeof(int));

        // Send chunks to ranks 1..5
        for (int r = 1; r < size; r++) {
            MPI_Send(&data[displs[r]], counts[r], MPI_INT, r, 0, MPI_COMM_WORLD);
        }

        // Rank 0 sorts its local chunk
        qsort(local, local_n, sizeof(int), compare_int);

        // Put rank0 sorted chunk back into data
        memcpy(&data[displs[0]], local, local_n * sizeof(int));

        // Receive sorted chunks back into the correct offsets
        for (int r = 1; r < size; r++) {
            MPI_Recv(&data[displs[r]], counts[r], MPI_INT, r, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }

        // Merge all sorted blocks in-place: merge block0 with block1, then with block2, etc.
        int merged_n = counts[0];
        for (int r = 1; r < size; r++) {
            merge(&data[0], merged_n, &data[displs[r]], counts[r]);
            merged_n += counts[r];
        }

        printf("Sorted:\t\t");
        for (int i = 0; i < NUM_ELEMENTS; i++) printf("%d ", data[i]);
        printf("\n");

    } else {
        // Other ranks receive their chunk
        MPI_Recv(local, local_n, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Sort locally
        qsort(local, local_n, sizeof(int), compare_int);

        // Send back sorted chunk
        MPI_Send(local, local_n, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    free(local);
    MPI_Finalize();
    return 0;
}