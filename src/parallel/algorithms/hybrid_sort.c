#include <mpi.h>
#include <stdlib.h>
#include "../algorithms.h"
#include "../utils.h"

double parallel_sample_sort(int r, int s, int* g, int curN) {
    int ln = curN / s;
    int* l = (int*)malloc(ln * sizeof(int));
    MPI_Scatter(g, ln, MPI_INT, l, ln, MPI_INT, 0, MPI_COMM_WORLD);

    double st = MPI_Wtime();

    /* Phase 0: local sort */
    qsort(l, ln, sizeof(int), cmp);

    /* Phase 1: pick s-1 equally-spaced samples */
    int ns = s - 1;  /* number of splitters */
    int* lsamp = (int*)malloc(ns * sizeof(int));
    for (int i = 0; i < ns; i++)
        lsamp[i] = l[(i + 1) * ln / s];

    /* Phase 2: rank 0 gathers, sorts, selects global splitters */
    int* asamp = NULL;
    if (!r) asamp = (int*)malloc(s * ns * sizeof(int));
    MPI_Gather(lsamp, ns, MPI_INT, asamp, ns, MPI_INT, 0, MPI_COMM_WORLD);

    int* gspl = (int*)malloc(ns * sizeof(int));
    if (!r) {
        qsort(asamp, s * ns, sizeof(int), cmp);
        for (int i = 0; i < ns; i++)
            gspl[i] = asamp[(i + 1) * ns];
        free(asamp);
    }

    /* Phase 3: broadcast global splitters */
    MPI_Bcast(gspl, ns, MPI_INT, 0, MPI_COMM_WORLD);

    /* Phase 4: partition l into s buckets using upper_bound */
    int* sc = (int*)calloc(s, sizeof(int));
    int* sd = (int*)calloc(s, sizeof(int));
    int* bounds = (int*)malloc((s + 1) * sizeof(int));
    bounds[0] = 0;
    for (int i = 0; i < ns; i++)
        bounds[i + 1] = upper_bound(l, ln, gspl[i]);
    bounds[s] = ln;
    for (int i = 0; i < s; i++) {
        sc[i] = bounds[i + 1] - bounds[i];
        sd[i] = bounds[i];
    }
    free(bounds);

    /* Phase 5: exchange counts then data */
    int* rc = (int*)malloc(s * sizeof(int));
    MPI_Alltoall(sc, 1, MPI_INT, rc, 1, MPI_INT, MPI_COMM_WORLD);

    int* rd = (int*)malloc(s * sizeof(int));
    rd[0] = 0;
    for (int i = 1; i < s; i++) rd[i] = rd[i - 1] + rc[i - 1];
    int tot = rd[s - 1] + rc[s - 1];
    int* rb = (int*)malloc(tot * sizeof(int));

    MPI_Alltoallv(l, sc, sd, MPI_INT, rb, rc, rd, MPI_INT, MPI_COMM_WORLD);

    /* Phase 6: re-sort received bucket */
    qsort(rb, tot, sizeof(int), cmp);

    double en = MPI_Wtime();

    /* Gather variable-sized results to rank 0 */
    int* rcs = NULL;
    int* dsp = NULL;
    if (!r) {
        rcs = (int*)malloc(s * sizeof(int));
        dsp = (int*)calloc(s, sizeof(int));
    }
    MPI_Gather(&tot, 1, MPI_INT, rcs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!r) {
        for (int i = 1; i < s; i++) dsp[i] = dsp[i - 1] + rcs[i - 1];
    }
    MPI_Gatherv(rb, tot, MPI_INT, g, rcs, dsp, MPI_INT, 0, MPI_COMM_WORLD);

    free(l); free(lsamp); free(gspl);
    free(sc); free(sd); free(rc); free(rd); free(rb);
    if (!r) { free(rcs); free(dsp); }

    return en - st;
}
