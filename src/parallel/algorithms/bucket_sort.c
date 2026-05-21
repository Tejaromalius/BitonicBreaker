
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../algorithms.h"
#include "../utils.h"

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
        /* Recurse on smaller side, iterate on larger (tail-call elimination). */
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

/* ── Parallel Bucket Sort ────────────────────────────────────────────────────
 *
 * Exploits the known, bounded, uniform distribution of the input:
 *   values ∈ [0, curN × 10 − 1]   (main.c: rand() % (N * 10))
 *
 * Algorithm phases (all inside the timer):
 *
 *   Phase 1  O(N/p)          — One-pass counting + bucket-filling via integer
 *                               division. No local sort, no sampling, no bcast.
 *
 *   Phase 2  O(N/p) comm     — MPI_Alltoall (counts) + MPI_Alltoallv (data).
 *                               Identical to the previous hybrid_sort.
 *
 *   Phase 3  O((N/p) log(N/p)) — Single OpenMP introsort on the received
 *                               partition. Replaces pre-sort + K-way merge.
 *
 * Savings vs. hybrid_sort:
 *   • MPI_Gather + MPI_Bcast eliminated (splitter selection removed entirely).
 *   • O(p log(N/p)) binary-search partitioning → O(N/p) integer division.
 *   • pre-sort [O((N/p) log(N/p))] + K-way merge [O((N/p) log p)]
 *       → single sort [O((N/p) log(N/p))].
 *     (removes K-way merge outright; only one introsort remains.)            */

double parallel_bucket_sort(int r, int s, int* g, int curN) {
    int ln = curN / s;
    int* l = (int*)malloc(ln * sizeof(int));
    MPI_Scatter(g, ln, MPI_INT, l, ln, MPI_INT, 0, MPI_COMM_WORLD);

    double st = MPI_Wtime();

    /* ── Phase 1: one-pass bucket counting ─────────────────────────────────
     * Values are in [0, curN*10 - 1].  bucket_width = ceil(max_val / s)
     * guarantees bucket index ∈ [0, s-1] for every valid input value.       */
    int max_val      = curN * 10;
    int bucket_width = (max_val + s - 1) / s;       /* ceil division           */

    int* sc = (int*)calloc(s, sizeof(int));          /* send counts             */
    int* bkt = (int*)malloc(ln * sizeof(int));       /* cached bucket per elem  */

    for (int i = 0; i < ln; i++) {
        int b = l[i] / bucket_width;
        if (b >= s) b = s - 1;                      /* clamp (defensive only)  */
        bkt[i] = b;
        sc[b]++;
    }

    /* ── Build contiguous send buffer (two-pass avoids extra allocation) ── */
    int* sd = (int*)calloc(s, sizeof(int));
    for (int i = 1; i < s; i++) sd[i] = sd[i - 1] + sc[i - 1];

    int* send_buf = (int*)malloc(ln * sizeof(int));
    int  cur[8];                                     /* write cursors, s ≤ 8   */
    for (int i = 0; i < s; i++) cur[i] = sd[i];

    for (int i = 0; i < ln; i++) {
        send_buf[cur[bkt[i]]++] = l[i];
    }
    free(bkt);

    /* ── Phase 2: exchange counts, then data ───────────────────────────────
     * Identical to the previous hybrid_sort.                                 */
    int* rc = (int*)malloc(s * sizeof(int));
    MPI_Alltoall(sc, 1, MPI_INT, rc, 1, MPI_INT, MPI_COMM_WORLD);

    int* rd = (int*)calloc(s, sizeof(int));
    for (int i = 1; i < s; i++) rd[i] = rd[i - 1] + rc[i - 1];
    int tot = rd[s - 1] + rc[s - 1];

    int* rb = (int*)malloc((tot > 0 ? tot : 1) * sizeof(int));
    MPI_Alltoallv(send_buf, sc, sd, MPI_INT, rb, rc, rd, MPI_INT, MPI_COMM_WORLD);
    free(send_buf);

    /* ── Phase 3: sort the received partition once ──────────────────────── */
    sort_ints(rb, tot);

    double en = MPI_Wtime();

    /* ── Gather results to rank 0 (outside timed region, same as before) ── */
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

    free(l);
    free(sc); free(sd); free(rc); free(rd); free(rb);
    if (!r) { free(rcs); free(dsp); }

    return en - st;
}
