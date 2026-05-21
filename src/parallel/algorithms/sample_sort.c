#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../algorithms.h"
#include "../utils.h"

/* --- Specialized integer sort: introsort with insertion-sort base case. ---
 * No function-pointer comparator, so the hot loop inlines fully. */

static inline void iswap(int* a, int* b) { int t = *a; *a = *b; *b = t; }

static void insertion_sort_int(int* a, int n) {
    for (int i = 1; i < n; i++) {
        int x = a[i], j = i - 1;
        while (j >= 0 && a[j] > x) { a[j + 1] = a[j]; j--; }
        a[j + 1] = x;
    }
}

/* Median-of-three pivot, then Hoare-style partition. */
static int partition_int(int* a, int n) {
    int mid = n / 2, lo = 0, hi = n - 1;
    if (a[lo] > a[mid]) iswap(&a[lo], &a[mid]);
    if (a[lo] > a[hi])  iswap(&a[lo], &a[hi]);
    if (a[mid] > a[hi]) iswap(&a[mid], &a[hi]);
    int pivot = a[mid];
    iswap(&a[mid], &a[hi - 1]);
    int i = lo, j = hi - 1;
    for (;;) {
        while (a[++i] < pivot) {}
        while (a[--j] > pivot) {}
        if (i >= j) break;
        iswap(&a[i], &a[j]);
    }
    iswap(&a[i], &a[hi - 1]);
    return i;
}

#define INSERTION_CUTOFF 32
#define OMP_TASK_CUTOFF  16384

static void introsort_serial(int* a, int n) {
    while (n > INSERTION_CUTOFF) {
        int p = partition_int(a, n);
        /* recurse on smaller side, iterate on larger (tail-call elimination) */
        if (p < n - p) {
            introsort_serial(a, p);
            a += p + 1; n -= p + 1;
        } else {
            introsort_serial(a + p + 1, n - p - 1);
            n = p;
        }
    }
    insertion_sort_int(a, n);
}

#ifdef _OPENMP
static void introsort_par(int* a, int n) {
    if (n <= OMP_TASK_CUTOFF) { introsort_serial(a, n); return; }
    int p = partition_int(a, n);
    #pragma omp task default(none) firstprivate(a, p) if(p > OMP_TASK_CUTOFF)
    introsort_par(a, p);
    introsort_par(a + p + 1, n - p - 1);
    #pragma omp taskwait
}
#endif

static void sort_ints(int* a, int n) {
    if (n <= 1) return;
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp single nowait
        introsort_par(a, n);
    }
#else
    introsort_serial(a, n);
#endif
}

/* --- k-way merge of `k` already-sorted runs into `out` using a tiny heap. ---
 * runs[i] points at the start of the i-th run, with length sizes[i].
 * Total of all sizes must equal `total`. */

typedef struct { int val; int run; } heap_node;

static inline void heap_sift_down(heap_node* h, int n, int i) {
    for (;;) {
        int l = 2 * i + 1, r = 2 * i + 2, s = i;
        if (l < n && h[l].val < h[s].val) s = l;
        if (r < n && h[r].val < h[s].val) s = r;
        if (s == i) return;
        heap_node t = h[i]; h[i] = h[s]; h[s] = t;
        i = s;
    }
}

static void kway_merge(int** runs, int* sizes, int k, int* out, int total) {
    /* fast paths */
    if (k == 1) { memcpy(out, runs[0], total * sizeof(int)); return; }
    if (k == 2) {
        int i = 0, j = 0, o = 0;
        int *a = runs[0], *b = runs[1];
        int na = sizes[0], nb = sizes[1];
        while (i < na && j < nb) out[o++] = (a[i] <= b[j]) ? a[i++] : b[j++];
        while (i < na) out[o++] = a[i++];
        while (j < nb) out[o++] = b[j++];
        return;
    }

    int* pos = (int*)calloc(k, sizeof(int));
    heap_node* heap = (heap_node*)malloc(k * sizeof(heap_node));
    int hn = 0;
    for (int i = 0; i < k; i++) {
        if (sizes[i] > 0) {
            heap[hn].val = runs[i][0];
            heap[hn].run = i;
            hn++;
            pos[i] = 1;
        }
    }
    /* build heap */
    for (int i = hn / 2 - 1; i >= 0; i--) heap_sift_down(heap, hn, i);

    int o = 0;
    while (hn > 0) {
        out[o++] = heap[0].val;
        int run = heap[0].run;
        if (pos[run] < sizes[run]) {
            heap[0].val = runs[run][pos[run]++];
            /* heap[0].run unchanged */
        } else {
            heap[0] = heap[--hn];
        }
        heap_sift_down(heap, hn, 0);
    }
    (void)total;
    free(pos);
    free(heap);
}

double parallel_sample_sort(int r, int s, int* g, int curN) {
    int ln = curN / s;
    int* l = (int*)malloc(ln * sizeof(int));
    MPI_Scatter(g, ln, MPI_INT, l, ln, MPI_INT, 0, MPI_COMM_WORLD);

    double st = MPI_Wtime();

    /* Phase 0: local sort — specialized integer introsort + OpenMP. */
    sort_ints(l, ln);

    /* Phase 1: pick s-1 equally-spaced samples. */
    int ns = s - 1;
    int* lsamp = NULL;
    int* asamp = NULL;
    int* gspl  = (int*)malloc((ns > 0 ? ns : 1) * sizeof(int));
    if (ns > 0) {
        lsamp = (int*)malloc(ns * sizeof(int));
        for (int i = 0; i < ns; i++) lsamp[i] = l[(i + 1) * ln / s];
        if (!r) asamp = (int*)malloc(s * ns * sizeof(int));
        MPI_Gather(lsamp, ns, MPI_INT, asamp, ns, MPI_INT, 0, MPI_COMM_WORLD);

        if (!r) {
            sort_ints(asamp, s * ns);
            for (int i = 0; i < ns; i++) gspl[i] = asamp[(i + 1) * ns];
            free(asamp);
        }
        MPI_Bcast(gspl, ns, MPI_INT, 0, MPI_COMM_WORLD);
    }

    /* Phase 2: partition local sorted array into s buckets via upper_bound. */
    int* sc = (int*)calloc(s, sizeof(int));
    int* sd = (int*)calloc(s, sizeof(int));
    int* bounds = (int*)malloc((s + 1) * sizeof(int));
    bounds[0] = 0;
    for (int i = 0; i < ns; i++) bounds[i + 1] = upper_bound(l, ln, gspl[i]);
    bounds[s] = ln;
    for (int i = 0; i < s; i++) {
        sc[i] = bounds[i + 1] - bounds[i];
        sd[i] = bounds[i];
    }
    free(bounds);

    /* Phase 3: exchange counts then data. */
    int* rc = (int*)malloc(s * sizeof(int));
    MPI_Alltoall(sc, 1, MPI_INT, rc, 1, MPI_INT, MPI_COMM_WORLD);

    int* rd = (int*)malloc(s * sizeof(int));
    rd[0] = 0;
    for (int i = 1; i < s; i++) rd[i] = rd[i - 1] + rc[i - 1];
    int tot = rd[s - 1] + rc[s - 1];
    int* rb = (int*)malloc((tot > 0 ? tot : 1) * sizeof(int));

    MPI_Alltoallv(l, sc, sd, MPI_INT, rb, rc, rd, MPI_INT, MPI_COMM_WORLD);

    /* Phase 4: k-way merge of the s already-sorted runs inside rb. */
    int* merged = (int*)malloc((tot > 0 ? tot : 1) * sizeof(int));
    int** runs = (int**)malloc(s * sizeof(int*));
    for (int i = 0; i < s; i++) runs[i] = rb + rd[i];
    kway_merge(runs, rc, s, merged, tot);
    free(runs);

    double en = MPI_Wtime();

    /* Gather variable-sized results to rank 0 (outside timed region). */
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
    MPI_Gatherv(merged, tot, MPI_INT, g, rcs, dsp, MPI_INT, 0, MPI_COMM_WORLD);

    free(l);
    if (lsamp) free(lsamp);
    free(gspl);
    free(sc); free(sd); free(rc); free(rd); free(rb); free(merged);
    if (!r) { free(rcs); free(dsp); }

    return en - st;
}
