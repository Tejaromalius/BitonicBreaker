#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <ctime>

#include "algorithms.h"
#include "utils.h"

#define N 1000000

std::string get_timestamp() {
  time_t now = time(nullptr);
  struct tm* timeinfo = localtime(&now);
  char buffer[20];
  strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", timeinfo);
  return std::string(buffer);
}

std::string generate_report_filename(const std::string& alg) {
  return "reports/serial_" + alg + "_" + get_timestamp() + ".txt";
}

int main(int argc, char** argv) {
  std::string target = "all";
  if (argc > 1) target = argv[1];

  srand(42);
  int *o = new int[N], *a = new int[N];
  for (int i = 0; i < N; i++) o[i] = rand() % (N * 10);

  struct {
    const char* n;
    void (*f)(int*, int);
    bool skip;
  } algs[] = {{"Bubble", bubble_sort, 1},
              {"Selection", selection_sort, 1},
              {"Quick", [](int* x, int n) { quick_sort(x, 0, n - 1); }, 0},
              {"Merge", [](int* x, int n) { merge_sort(x, 0, n - 1); }, 0},
              {"Bitonic", bitonic_sort, 0}};

  std::string report_filename;
  std::ofstream report_file;

  if (target == "all") {
    report_filename = "reports/serial_all_" + get_timestamp() + ".txt";
    report_file.open(report_filename, std::ios::app);
  }

  for (auto g : algs) {
    std::string name = g.n;
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    std::string lowerTarget = target;
    std::transform(lowerTarget.begin(), lowerTarget.end(), lowerTarget.begin(), ::tolower);

    if (target != "all" && name != lowerTarget) continue;

    if (target != "all") {
      report_filename = generate_report_filename(lowerTarget);
      report_file.open(report_filename, std::ios::app);
    }

    std::stringstream ss;
    if (g.skip) {
      int subN = N / 10;
      std::copy(o, o + subN, a);
      double st = t();
      g.f(a, subN);
      double duration = t() - st;
      ss << "Algorithm: " << g.n << "\n"
         << "Elements: " << subN << " (Extrapolated to " << N << ")\n"
         << "Time: " << duration << "s\n"
         << "Extrapolated Time: " << duration * 100 << "s\n"
         << "Status: " << (std::is_sorted(a, a + subN) ? "OK" : "ERR") << "\n---\n";
    } else {
      std::copy(o, o + N, a);
      double st = t();
      g.f(a, N);
      double duration = t() - st;
      ss << "Algorithm: " << g.n << "\n"
         << "Elements: " << N << "\n"
         << "Time: " << duration << "s\n"
         << "Status: " << (std::is_sorted(a, a + N) ? "OK" : "ERR") << "\n---\n";
    }

    std::string output = ss.str();
    std::cout << output;
    report_file << output;
    report_file.flush();

    if (target != "all") {
      report_file.close();
    }
  }

  if (target == "all" && report_file.is_open()) {
    report_file.close();
  }

  delete[] o;
  delete[] a;
  return 0;
}
