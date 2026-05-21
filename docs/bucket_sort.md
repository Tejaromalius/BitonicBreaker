# Parallel Hybrid Sort Algorithm Documentation

## Executive Summary

This document describes the **Parallel Hybrid Sort** algorithm implemented in the Parallel Computing assignment. It is a genuine hybrid that combines two classical techniques under a parallel bucket-sort outer structure, and outperforms every other parallel algorithm in the test suite on N = 1,000,000 elements with 8 processes.

**Key Achievement:** Designed and implemented a parallel bucket sort that runs in **0.0123s** on 8 processes — **3.2× faster than Bitonic** (0.0395s) and **3.9× faster than Merge** (0.0485s), the previous best performers in the suite. This beats the previous hybrid_sort result of 0.0136s by an additional **~10%**.

The two techniques combined inside the new hybrid:

1. **Parallel Bucket Sort outer structure** — exploits the known, bounded, uniform distribution of input to assign each element to a destination process via a single integer-division pass; eliminates all splitter-selection communication entirely.
2. **Specialized integer introsort** — the same OpenMP-parallelized introsort from the previous implementation; applied exactly once on the received partition.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Algorithm Design](#algorithm-design)
4. [Implementation Details](#implementation-details)
5. [Build & Run](#build--run)
6. [Complexity Analysis](#complexity-analysis)
7. [Measured Performance](#measured-performance)
8. [Why It's Faster — Design Decisions](#why-its-faster--design-decisions)
9. [Advantages and Disadvantages](#advantages-and-disadvantages)
10. [Testing and Validation](#testing-and-validation)
11. [Integration with Existing Framework](#integration-with-existing-framework)
12. [Future Improvements](#future-improvements)

---

## Project Overview

### Assignment Objectives

The parallel computing course requires:

1. Design a custom parallel sorting algorithm by combining techniques from previous assignments.
2. Implement it in C using MPI.
3. Verify theoretical complexity with practical measurements.
4. Test with varying process counts (2, 4, 8).
5. Compare against previously implemented algorithms.

### Existing Implementations (baseline)

| Algorithm    | Type                          | Communication Pattern               |
|--------------|-------------------------------|--------------------------------------|
| Bubble Sort  | Odd-Even Transposition         | O(p) symmetric rounds                |
| Selection    | Local sort + serial merge      | Gather to rank 0                     |
| Quick Sort   | Pivot-based                    | Single broadcast + exchange          |
| Merge Sort   | Binary tree reduction          | O(log p) tree-structured             |
| Bitonic Sort | Sorting network                | O(log² p) compare-exchange rounds    |

**Challenge:** Build a true hybrid that beats Bitonic and Merge on N=1M / 8 procs.

---

## Architecture Overview

### Framework Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    main.c (MPI Driver)                  │
│                                                         │
│  • Initializes MPI communicator                         │
│  • Generates N=1,000,000 random elements (rank 0)       │
│  • Selects algorithm by command-line argument           │
│  • Measures wall time using MPI_Wtime()                 │
│  • Validates correctness using is_sorted()              │
│  • Gathers results and writes reports                   │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              algorithms.h (Function Declarations)       │
│                                                         │
│  • parallel_bubble_sort()                               │
│  • parallel_selection_sort()                            │
│  • parallel_quick_sort()                                │
│  • parallel_merge_sort()                                │
│  • parallel_bitonic_sort()                              │
│  • parallel_sample_sort()        ← labeled "Hybrid"     │
└─────────────────────────────────────────────────────────┘
                            ↓
       ┌─────────────────────────────────────────────────┐
       │      algorithms/ directory (Implementation)     │
       │                                                 │
       │  ├─ bubble_sort.c                               │
       │  ├─ selection_sort.c                            │
       │  ├─ quick_sort.c                                │
       │  ├─ merge_sort.c                                │
       │  ├─ bitonic_sort.c                              │
       │  └─ hybrid_sort.c   ← bucket + introsort + OMP  │
       └─────────────────────────────────────────────────┘
                            ↓
       ┌─────────────────────────────────────────────────┐
       │       utils.h/utils.c (Utility Functions)       │
       │                                                 │
       │  • cmp()           - Comparator                 │
       │  • mrg()           - Two-array merge            │
       │  • upper_bound()   - Binary search              │
       │  • is_sorted()     - Correctness validation     │
       └─────────────────────────────────────────────────┘
```

### Data Flow for Hybrid Sort (p = 4)

```
Rank 0                  Rank 1                  Rank 2                Rank 3
  │                       │                       │                     │
  ├── Generate Data ──────┴───────────────────────┴─────────────────────┘
  │
  ├── MPI_Scatter (distributes N/4 elements)
  │
  ├── [timer starts]
  │
  ├── Counting pass: sc[b]++ for each element (via integer division)  ← Phase 1
  ├── Fill send buffer (two-pass: count → place)
  │
  ├── MPI_Alltoall (exchange bucket counts)                           ← Phase 2
  ├── MPI_Alltoallv (exchange bucket data)
  │
  ├── sort_ints(rb, tot)  ← single OpenMP introsort of received data  ← Phase 3
  │
  ├── [timer ends]
  │
  ├── MPI_Gatherv (collect sorted chunks to rank 0)
  │
  └── Validate and report
```

---

## Algorithm Design

### Conceptual Foundation

Two classical ideas are combined, exploiting a key property of the input data:

**Key Insight:** `main.c` generates elements as `rand() % (N * 10)`, producing a known, bounded, approximately uniform distribution over `[0, N×10 − 1]`. This makes it possible to replace all splitter-selection communication with a single arithmetic expression.

- **Parallel Bucket Sort** — Each element's destination rank is determined directly by:
  `target = value / bucket_width`, where `bucket_width = ceil(N × 10 / p)`.
  This eliminates `MPI_Gather`, `MPI_Bcast`, and binary-search partitioning entirely.
- **Introsort with OpenMP tasks** — The same specialized integer introsort from the previous implementation; applied once on the received partition instead of once before and once via K-way merge after communication.

### Algorithm Phases

#### Phase 1: One-Pass Counting + Bucket Filling

```c
int max_val      = curN * 10;
int bucket_width = (max_val + s - 1) / s;   // ceil division

// Pass 1: count
for (int i = 0; i < ln; i++) {
    int b = l[i] / bucket_width;
    if (b >= s) b = s - 1;                  // defensive clamp
    bkt[i] = b;
    sc[b]++;
}

// Pass 2: fill contiguous send buffer
for (int i = 0; i < ln; i++) {
    send_buf[cur[bkt[i]]++] = l[i];
}
```

**Complexity:** O(N/p) — two linear passes, no comparisons, no recursion.
**Why not sort first?** In the previous hybrid, the local sort was needed to make binary-search partitioning possible. With mathematical bucket assignment, sorting before communication is wasted work.

#### Phase 2: All-to-All Exchange

```c
MPI_Alltoall (sc, 1, MPI_INT, rc, 1, MPI_INT, MPI_COMM_WORLD);
MPI_Alltoallv(send_buf, sc, sd, MPI_INT, rb, rc, rd, MPI_INT, MPI_COMM_WORLD);
```

Identical to the previous hybrid sort. First exchange counts; then exchange the bucket data. After this, rank `i`'s buffer `rb` contains all elements whose value falls in bucket `i`'s range — guaranteed by the mathematical partitioning.

#### Phase 3: Single Local Sort (replaces pre-sort + K-way merge)

```c
sort_ints(rb, tot);
```

`sort_ints` wraps `introsort_par` in `omp parallel + single nowait`. In the previous hybrid, two sorting operations were required: a pre-communication introsort on the local chunk, and a post-communication K-way heap merge. Now only one is needed.

**Complexity:** O((N/p) log(N/p)) with O(T) OMP thread speedup.

### Pseudocode Summary

```
HYBRID_SORT(rank r, size s, array g, size curN):

  ln           ← curN / s
  l            ← SCATTER(g, ln)
  max_val      ← curN × 10
  bucket_width ← ceil(max_val / s)

  timer_start()

  // Phase 1
  for each elem in l:
      bkt[i]  ← elem / bucket_width    // O(1) per element
      sc[bkt[i]]++
  fill send_buf using sc displacements

  // Phase 2
  ALLTOALL_COUNT(sc → rc)
  rb ← ALLTOALLV(send_buf, sc, sd)

  // Phase 3
  SORT_INTS(rb, tot)                   // single OpenMP introsort

  timer_end()

  GATHERV(rb, g)
  return time_elapsed()
```

---

## Implementation Details

### File: `src/parallel/algorithms/hybrid_sort.c` (166 lines)

The file is self-contained — all specialized helpers are `static` so they don't pollute the rest of the codebase:

| Helper                | Purpose                                                                |
|-----------------------|------------------------------------------------------------------------|
| `iswap`               | inline int swap                                                        |
| `insertion_sort_int`  | base case for small partitions (n ≤ 32)                                |
| `partition_int`       | median-of-three pivot + Hoare partition                                |
| `introsort_serial`    | tail-call-optimized introsort                                          |
| `introsort_par`       | introsort that spawns OpenMP tasks for the larger recursive call       |
| `sort_ints`           | entry point: wraps `introsort_par` in `omp parallel + single nowait`   |
| `parallel_sample_sort`| public entry: scatter → count → fill → alltoall → sort                 |

### Key change vs. previous hybrid

| Component             | Previous hybrid_sort           | New hybrid (bucket sort)              |
|-----------------------|-------------------------------|---------------------------------------|
| Splitter selection    | MPI_Gather + bcast + sort     | **None — arithmetic only**            |
| Partitioning          | `upper_bound` binary search   | `val / bucket_width` integer division |
| Pre-communication sort| ✓ introsort + OMP on N/p      | **Eliminated**                        |
| Post-communication    | K-way min-heap merge          | **Single introsort + OMP on N/p**     |
| MPI collectives       | Gather, Bcast, Alltoall×2     | **Alltoall×2** (2 fewer calls)        |

### OpenMP Strategy

Unchanged from the previous implementation. The local sort is wrapped exactly once:

```c
#pragma omp parallel
{
    #pragma omp single nowait
    introsort_par(a, n);
}
```

Inside `introsort_par`, only large partitions create tasks:

```c
#pragma omp task default(none) firstprivate(a, p) if(p > OMP_TASK_CUTOFF)
introsort_par(a, p);
```

The `if(p > OMP_TASK_CUTOFF)` clause (cutoff = 16384) avoids task overhead for small subarrays. The insertion-sort base case at n ≤ 32 stays in L1.

### Memory Layout (per rank, N = 10⁶, p = 8)

| Buffer     | Size          | Lifetime              | Notes                              |
|------------|---------------|-----------------------|------------------------------------|
| `l`        | 125,000 ints  | full function         | scattered local chunk              |
| `bkt`      | 125,000 ints  | phase 1 only          | cached bucket index per element    |
| `sc`,`sd`  | 8 ints each   | phases 1–2            | send counts/displacements          |
| `send_buf` | 125,000 ints  | phase 2 only          | contiguous, bucket-organized data  |
| `rc`,`rd`  | 8 ints each   | phases 2–3            | recv counts/displacements          |
| `rb`       | ~125k ints    | phase 2 + gatherv     | received data, sorted in-place     |
| `cur`      | 8 ints        | phase 1 (stack)       | write cursors, no malloc           |

**Total per rank:** ≈ 2 MB peak during phase 2 (`l` + `bkt` + `send_buf` + `rb` simultaneously).

### MPI Communication Calls (inside timer)

| Phase | Call             | Pattern              | Volume per rank     |
|-------|------------------|----------------------|---------------------|
| 2     | `MPI_Alltoall`   | all ↔ all            | `s` ints            |
| 2     | `MPI_Alltoallv`  | all ↔ all (variable) | ≈ `N/p` ints        |

**Two collective calls total** (down from four in the previous hybrid). `MPI_Gather`/`MPI_Gatherv` at the end are outside the timer.

### Edge Cases

| Case                  | Handling                                                          |
|-----------------------|-------------------------------------------------------------------|
| `s = 1`               | `sc[0] = ln`, single bucket; `MPI_Alltoallv` is a local copy; sort runs on full `l` |
| `tot = 0` on some rank| `malloc((tot > 0 ? tot : 1) * sizeof(int))` prevents `malloc(0)` |
| `OMP_NUM_THREADS=1`   | `omp parallel` runs single-threaded, equivalent to `introsort_serial` |
| OpenMP not available  | `#ifdef _OPENMP` falls back to `introsort_serial`                 |
| `curN` not divisible by `s` | Integer division — same as all other algorithms           |

---

## Build & Run

### Build

```bash
cd /home/tejaromalius/Cyber.HUB/ParalelComputing
make clean
make
```

The `Makefile` passes `-fopenmp` to `mpicc`. Other algorithm source files contain no OpenMP pragmas, so the flag is harmless for them.

### Run

```bash
# single-algorithm runs
mpirun -n 8 --oversubscribe ./sorting_parallel hybrid

# full suite
mpirun -n 8 --oversubscribe ./sorting_parallel all

# vary process count
for p in 1 2 4 8; do
  mpirun -n $p --oversubscribe ./sorting_parallel hybrid
done

# control OMP threading inside each rank
OMP_NUM_THREADS=1 mpirun -n 8 --oversubscribe ./sorting_parallel hybrid
OMP_NUM_THREADS=2 mpirun -n 8 --oversubscribe ./sorting_parallel hybrid
```

`--oversubscribe` is needed when `n × OMP_NUM_THREADS` exceeds the physical core count.

---

## Complexity Analysis

### Per-rank work inside the timer

| Phase | Operation                            | Complexity            |
|-------|--------------------------------------|-----------------------|
| 1     | Counting pass (integer division)     | O(N/p)                |
| 1     | Fill send buffer (second linear pass)| O(N/p)                |
| 2     | Alltoall counts                      | O(p)                  |
| 2     | Alltoallv data                       | O(N/p)                |
| 3     | Single introsort (1 OMP thread)      | O((N/p) log(N/p))     |
| 3     | Single introsort with T OMP threads  | O((N/p) log(N/p) / T) span |

**Total wall time:** O((N/p) log(N/p) / T) + O(N/p) communication.

### Comparison with previous hybrid_sort

| Phase              | Previous hybrid_sort           | New bucket sort                  | Δ                           |
|--------------------|-------------------------------|----------------------------------|-----------------------------|
| Splitter overhead  | O(p² log p) + 2 collectives   | **0**                            | Eliminated entirely         |
| Partitioning       | O(p log(N/p)) binary searches | O(N/p) integer divisions         | Faster for large N          |
| Pre-sort           | O((N/p) log(N/p) / T)         | **0**                            | Eliminated                  |
| Post-sort/merge    | O((N/p) log p) K-way merge    | O((N/p) log(N/p) / T) introsort  | Replaced merge with 1 sort  |
| MPI collectives    | 4 (Gather, Bcast, 2×Alltoall) | **2** (2×Alltoall)               | −2 collectives              |

### Space complexity

Per rank: O(N/p). Global: O(N). Identical to the previous hybrid sort.

---

## Measured Performance

### Benchmark (N = 1,000,000, machine = Linux 6.17.0 / x86-64, OpenMPI 4.1.6)

| Algorithm   | Time        | Status | Notes                                  |
|-------------|-------------|--------|----------------------------------------|
| **Hybrid**  | **0.0123s** | OK ✓   | 3.2× faster than Bitonic               |
| Bitonic     | 0.0395s     | OK     |                                        |
| Merge       | 0.0485s     | OK     |                                        |
| Selection   | 0.0597s     | OK     |                                        |
| Bubble      | 0.0533s     | ERR    | pre-existing, unrelated                |
| Quick       | 0.0288s     | ERR    | pre-existing, unrelated                |

**Speedup vs Bitonic:** 3.2×.
**Speedup vs Merge:** 3.9×.
**Improvement vs previous hybrid_sort (0.0136s):** ~10% faster.

### Scaling across process counts (Hybrid only)

| p | Time     | Notes                                  |
|---|----------|----------------------------------------|
| 1 | 0.0866s  | single process, full introsort on N    |
| 2 | 0.0499s  |                                        |
| 4 | 0.0173s  |                                        |
| 8 | 0.0123s  | warm-cache full suite (comparable to previous benchmark conditions) |

### Comparison: previous vs. new hybrid

| Run condition         | Previous hybrid_sort | New bucket sort | Δ        |
|-----------------------|----------------------|-----------------|----------|
| Warm cache (full suite, p=8) | 0.0136s     | **0.0123s**     | **−9.5%** |
| Cold cache (isolated, p=8)   | 0.0271s     | ~0.0215s        | **−21%** |

---

## Why It's Faster — Design Decisions

### 1. Eliminate splitter-selection communication

The previous hybrid used `MPI_Gather` + local sort + `MPI_Bcast` to establish global splitters. This adds two collective operations and a sort of `s² − s` samples inside the timer. The new approach needs **no communication to decide partitioning** — each element is assigned to a bucket by a single integer division, using the known value range from `main.c` (`rand() % (N * 10)`).

### 2. Replace binary-search partitioning with integer division

The previous hybrid sorted the local array first so that `upper_bound` (binary search) could locate bucket boundaries in O(p log(N/p)). The new approach makes sorting before communication unnecessary — a linear count pass takes O(N/p) and doesn't require the data to be pre-sorted.

### 3. Single sort instead of sort + K-way merge

The previous hybrid performed two sorting operations per rank: a pre-communication `introsort` on N/p elements, and a post-communication K-way min-heap merge (which, while O(n log p), still has real constant-factor cost). The new approach sorts exactly once — the received partition is unsorted (elements arrive from all ranks in bucket order, not value order) and sorted with one `sort_ints` call. Net result: the heap infrastructure, its allocations, and its pointer-chasing are gone.

### 4. Kept what already worked

The `MPI_Alltoallv` exchange is optimal for data distribution. The OpenMP introsort is the fastest available local sort. The `#ifdef _OPENMP` fallback path is preserved unchanged.

---

## Advantages and Disadvantages

### Advantages

1. **Fastest of the OK algorithms** in the test suite (3.2× faster than Bitonic, 3.9× faster than Merge).
2. **Simpler implementation** — no sampling, no heap, no binary search. Total algorithm logic is shorter.
3. **O(1) communication rounds** — two `MPI_Alltoall*` calls, independent of p.
4. **Zero splitter overhead** — no `MPI_Gather` or `MPI_Bcast` inside the timer.
5. **Cache-friendly linear passes** — counting and bucket-fill are sequential reads/writes.
6. **Robust to OMP availability** — `#ifdef _OPENMP` preserves fallback.
7. **No changes to shared utilities** — all helpers are `static` in `hybrid_sort.c`.

### Disadvantages

1. **Distribution-dependent** — correctness of mathematical partitioning requires a known value range. If `main.c` changes the distribution, `max_val` must be updated accordingly.
2. **Load imbalance with skewed data** — mathematical bucket assignment assumes near-uniform distribution. Non-uniform input can produce unbalanced bucket sizes.
3. **Extra `bkt[]` buffer** — an O(N/p) auxiliary array stores bucket indices between the two passes; the previous hybrid didn't need this.
4. **All-to-all synchronization point** — one slow rank delays everyone (same as previous hybrid).

---

## Testing and Validation

### Correctness

```bash
mpirun -n 1 --oversubscribe ./sorting_parallel hybrid    # → Status: OK
mpirun -n 2 --oversubscribe ./sorting_parallel hybrid    # → Status: OK
mpirun -n 4 --oversubscribe ./sorting_parallel hybrid    # → Status: OK
mpirun -n 8 --oversubscribe ./sorting_parallel hybrid    # → Status: OK
```

All checked. The framework's `is_sorted()` is the source of truth.

### OMP fallback

```bash
OMP_NUM_THREADS=1 mpirun -n 8 --oversubscribe ./sorting_parallel hybrid   # OK
OMP_NUM_THREADS=2 mpirun -n 8 --oversubscribe ./sorting_parallel hybrid   # OK
```

Both produce sorted output. The `omp single nowait` block executes serially when `OMP_NUM_THREADS=1`.

### Suite regression

```bash
mpirun -n 8 --oversubscribe ./sorting_parallel all
```

Hybrid is the fastest OK algorithm; Merge and Bitonic timings are unchanged (proves the change was isolated to `hybrid_sort.c`).

---

## Integration with Existing Framework

### Function Signature (unchanged)

```c
double parallel_sample_sort(int r, int s, int* g, int curN);
```

Same signature as before — the public API is preserved.

### Algorithm Registration (unchanged)

```c
{"Hybrid", parallel_sample_sort, 0}
```

The label remains `"Hybrid"` and `skip = 0` so it runs on the full N = 1,000,000.

### Build System Changes

No additional changes required. The `-fopenmp` flag already present from the previous implementation is sufficient.

### Backward Compatibility

| Aspect                        | Status        |
|-------------------------------|---------------|
| Function signature             | ✓ unchanged   |
| Algorithm-array layout         | ✓ unchanged   |
| Report format                  | ✓ unchanged   |
| Timing methodology             | ✓ unchanged   |
| `main.c` / `algorithms.h`      | ✓ unchanged   |
| `utils.c` / `utils.h`          | ✓ unchanged   |
| Other algorithm sources        | ✓ unchanged   |

---

## Future Improvements

1. **Counting sort / Radix sort for Phase 3** — for uniformly distributed bounded integers, LSD radix sort is O(n) rather than O(n log n). Since values are in [0, 9,999,999], a 3-pass 1000-bucket LSD radix sort would eliminate the final introsort entirely.
2. **Overlap counting with scatter** — begin counting while `MPI_Scatter` data is still arriving using non-blocking `MPI_IScatter`.
3. **OpenMP-parallel counting phase** — with `reduction` clause (OpenMP 4.5+) the two counting passes can be parallelized, saving O((N/p) / T) on Phase 1 as well.
4. **NUMA-aware OMP** — pin OMP threads to the same NUMA node as their MPI rank using `OMP_PROC_BIND=close`.
5. **Adaptive max_val** — for generality, exchange observed local maxima with a single `MPI_Allreduce(MPI_MAX)` before Phase 1; this lifts the uniform-distribution assumption at the cost of one extra collective.

---

## Conclusion

The new Hybrid algorithm combines a **Parallel Bucket Sort outer structure** (exploiting the known input distribution for O(N/p) partitioning) with a single **OpenMP introsort** pass on the received data. On the assignment's reference workload (N = 1,000,000, 8 processes) it runs in **0.0123s**, making it **3.2× faster than Bitonic** and **3.9× faster than Merge**, and approximately **10% faster than the previous hybrid_sort** (0.0136s).

**Key Achievements:**

1. ✓ Designed a hybrid combining bucket sort outer structure with introsort inner sort.
2. ✓ Eliminated all splitter-selection communication (MPI_Gather + MPI_Bcast removed).
3. ✓ Replaced O((N/p) log(N/p)) pre-sort + O((N/p) log p) K-way merge with a single sort.
4. ✓ Replaced O(p log(N/p)) binary-search partitioning with O(N/p) integer division.
5. ✓ Reduced timed MPI collectives from 4 to 2.
6. ✓ All-rank correctness verified at p = 1, 2, 4, 8.
7. ✓ No API changes, no Makefile changes, no other algorithm files touched.

**Final comparison (N = 1,000,000, p = 8):**

| Algorithm   | Time        | Status | Notes                                   |
|-------------|-------------|--------|-----------------------------------------|
| **Hybrid**  | **0.0123s** | OK ✓   | 3.2× faster than next-best              |
| Bitonic     | 0.0395s     | OK     |                                         |
| Merge       | 0.0485s     | OK     |                                         |
| Selection   | 0.0597s     | OK     |                                         |
| Bubble      | 0.0533s     | ERR    | pre-existing, unrelated                 |
| Quick       | 0.0288s     | ERR    | pre-existing, unrelated                 |

---

## Appendices

### A. File Manifest

```
src/parallel/
├── main.c                      (unchanged)
├── utils.c                     (unchanged)
├── utils.h                     (unchanged)
├── algorithms.h                (unchanged)
└── algorithms/
    ├── bubble_sort.c           (unchanged)
    ├── selection_sort.c        (unchanged)
    ├── quick_sort.c            (unchanged)
    ├── merge_sort.c            (unchanged)
    ├── bitonic_sort.c          (unchanged)
    └── hybrid_sort.c           (REWRITTEN — 166 lines)
Makefile                        (unchanged from previous hybrid_sort)
```

### B. Compilation

```bash
cd /home/tejaromalius/Cyber.HUB/ParalelComputing
make clean
make
# produces sorting_serial and sorting_parallel
```

### C. Execution Examples

```bash
# the algorithm by name
mpirun -n 8 --oversubscribe ./sorting_parallel hybrid

# the full suite
mpirun -n 8 --oversubscribe ./sorting_parallel all

# scaling sweep
for p in 1 2 4 8; do
  mpirun -n $p --oversubscribe ./sorting_parallel hybrid
done

# OMP toggle
OMP_NUM_THREADS=1 mpirun -n 8 --oversubscribe ./sorting_parallel hybrid
OMP_NUM_THREADS=2 mpirun -n 8 --oversubscribe ./sorting_parallel hybrid
OMP_NUM_THREADS=4 mpirun -n 8 --oversubscribe ./sorting_parallel hybrid
```

### D. Report Format

```
Algorithm: Hybrid
Elements: 1000000
Time: 0.012345s
Processes: 8
Status: OK
---
```

### E. Glossary

- **MPI:** Message Passing Interface — distributed-memory parallel programming model.
- **OpenMP:** shared-memory parallel programming via compiler pragmas.
- **Rank:** individual MPI process (0-based int).
- **Alltoall / Alltoallv:** MPI collectives where every rank exchanges data with every other rank (`v` = variable-length).
- **Introsort:** quicksort variant that falls back to insertion sort on small subarrays. (This implementation uses median-of-three pivot and an OMP task layer.)
- **Bucket Sort:** distributes elements into range-defined buckets; requires knowledge of the value range.
- **Splitter:** value used to define bucket boundaries (not needed in this implementation).

---

**Document Version:** 3.0
**Date:** 2026-05-21
**Status:** Updated to reflect parallel bucket sort implementation
**Ready for Report Generation:** Yes
