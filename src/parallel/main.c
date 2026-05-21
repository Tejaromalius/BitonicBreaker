#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <ctype.h>

#include "algorithms.h"
#include "utils.h"

#define N 1000000

char* get_timestamp() {
  static char buffer[20];
  time_t now = time(NULL);
  struct tm* timeinfo = localtime(&now);
  strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", timeinfo);
  return buffer;
}

char* generate_report_filename(const char* alg, int processes) {
  static char filename[100];
  snprintf(filename, sizeof(filename), "reports/parallel_%s_%dp_%s.txt",
           alg, processes, get_timestamp());
  return filename;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int r, s;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  MPI_Comm_size(MPI_COMM_WORLD, &s);

  char *target = "all";
  if (argc > 1) target = argv[1];

  int *g = NULL, *o = NULL;
  if (!r) {
    g = (int*)malloc(N * sizeof(int));
    o = (int*)malloc(N * sizeof(int));
    srand(42);
    for (int i = 0; i < N; i++) o[i] = rand() % (N * 10);
  }

  struct {
    const char* n;
    double (*f)(int, int, int*, int);
    int skip;
  } algs[] = {{"Bubble", parallel_bubble_sort, 1},
              {"Selection", parallel_selection_sort, 1},
              {"Quick", parallel_quick_sort, 0},
              {"Merge", parallel_merge_sort, 0},
              {"Bitonic", parallel_bitonic_sort, 0},
              {"Sample", parallel_sample_sort, 0}};

  FILE *report_file = NULL;
  char report_filename[100];

  if (!r && strcmp(target, "all") == 0) {
    snprintf(report_filename, sizeof(report_filename), "reports/parallel_all_%dp_%s.txt",
             s, get_timestamp());
    report_file = fopen(report_filename, "a");
  }

  for (int i = 0; i < 6; i++) {
    char name[20];
    strcpy(name, algs[i].n);
    for(int j = 0; name[j]; j++) name[j] = tolower(name[j]);

    char target_lower[20];
    strcpy(target_lower, target);
    for(int j = 0; target_lower[j]; j++) target_lower[j] = tolower(target_lower[j]);

    if (strcmp(target, "all") != 0 && strcmp(name, target_lower) != 0) continue;

    int curN = N;
    if (algs[i].skip) curN = N / 10;

    if (!r) {
      memcpy(g, o, curN * sizeof(int));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double tl = algs[i].f(r, s, (!r ? g : NULL), curN);
    double tg;
    MPI_Reduce(&tl, &tg, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (!r) {
      if (strcmp(target, "all") != 0) {
        strcpy(report_filename, generate_report_filename(target_lower, s));
        report_file = fopen(report_filename, "a");
      }

      if (algs[i].skip) {
        printf("Algorithm: %s\n", algs[i].n);
        printf("Elements: %d (Extrapolated to %d)\n", curN, N);
        printf("Time: %fs\n", tg);
        printf("Extrapolated Time: %fs\n", tg * 100.0);
        printf("Processes: %d\n", s);
        printf("Status: %s\n---\n", is_sorted(g, curN) ? "OK" : "ERR");

        fprintf(report_file, "Algorithm: %s\n", algs[i].n);
        fprintf(report_file, "Elements: %d (Extrapolated to %d)\n", curN, N);
        fprintf(report_file, "Time: %fs\n", tg);
        fprintf(report_file, "Extrapolated Time: %fs\n", tg * 100.0);
        fprintf(report_file, "Processes: %d\n", s);
        fprintf(report_file, "Status: %s\n---\n", is_sorted(g, curN) ? "OK" : "ERR");
      } else {
        printf("Algorithm: %s\n", algs[i].n);
        printf("Elements: %d\n", curN);
        printf("Time: %fs\n", tg);
        printf("Processes: %d\n", s);
        printf("Status: %s\n---\n", is_sorted(g, curN) ? "OK" : "ERR");

        fprintf(report_file, "Algorithm: %s\n", algs[i].n);
        fprintf(report_file, "Elements: %d\n", curN);
        fprintf(report_file, "Time: %fs\n", tg);
        fprintf(report_file, "Processes: %d\n", s);
        fprintf(report_file, "Status: %s\n---\n", is_sorted(g, curN) ? "OK" : "ERR");
      }

      fflush(report_file);

      if (strcmp(target, "all") != 0) {
        fclose(report_file);
        report_file = NULL;
      }
    }
  }

  if (!r) {
    if (report_file) fclose(report_file);
    free(g);
    free(o);
  }
  MPI_Finalize();
  return 0;
}
