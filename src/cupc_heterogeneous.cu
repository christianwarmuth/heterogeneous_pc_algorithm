/*
 *	Adapted from https://github.com/LIS-Laboratory/cupc
 *
 */

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

#include <math.h>
#include <omp.h>

#include "constraint_heterogeneous.hpp"
#include "cupc_heterogeneous.h"

class HeterogeneousExecutionManager {
 public:
  explicit HeterogeneousExecutionManager(bool cuda_use_managed_memory, int* cuda_adjacency_matrix,
                                         int* adjacency_matrix, double* correlation_matrix, double* thresholds,
                                         int num_rows, int n)
      : cuda_use_managed_memory_(cuda_use_managed_memory),
        cuda_adjacency_matrix_(cuda_adjacency_matrix),
        adjacency_matrix_(adjacency_matrix),
        thresholds_(thresholds),
        num_rows_(num_rows),
        n_(n),
        skeleton_graph_(correlation_matrix, n),
        p_max_values_(n, std::vector<double>(n, -1.0)),
        separation_sets_(n, std::vector<std::vector<int>>(n, std::vector<int>({-1}))) {}

  void CalculateLevel(int level) {
    if (level == 0) {
      skeleton_graph_.CreateLevelZero(thresholds_[level], &p_max_values_, &separation_sets_, num_rows_, n_);
    } else {
      skeleton_graph_.FitToLevel(level, thresholds_[level], &p_max_values_, &separation_sets_, num_rows_, n_);
    }
  }

  void MergeLevel(int level) {
    if (cuda_use_managed_memory_) {
      std::swap(cuda_adjacency_matrix_, adjacency_matrix_);
    } else {
      cudaMemcpy(adjacency_matrix_, cuda_adjacency_matrix_, n_ * n_ * sizeof(int), cudaMemcpyDeviceToHost);
    }

    if (level == 0) {
      for (int i = 0; i < num_rows_; ++i) {
        for (int j = i + 1; j < n_; ++j) {
          if (skeleton_graph_.HasEdge(i, j)) {
            adjacency_matrix_[i * n_ + j] = 1;
            adjacency_matrix_[j * n_ + i] = 1;
          }
        }
      }
      for (int i = num_rows_; i < n_; ++i) {
        for (int j = i + 1; j < n_; ++j) {
          if (adjacency_matrix_[i * n_ + j] == 1) {
            skeleton_graph_.AddEdge(i, j);
          }
        }
      }
    } else {
      for (int i = 0; i < num_rows_; ++i) {
        for (int j = i + 1; j < n_; ++j) {
          if (!skeleton_graph_.HasEdge(i, j)) {
            adjacency_matrix_[i * n_ + j] = 0;
            adjacency_matrix_[j * n_ + i] = 0;
          }
        }
      }
      for (int i = num_rows_; i < n_; ++i) {
        for (int j = i + 1; j < n_; ++j) {
          if (adjacency_matrix_[i * n_ + j] == 0) {
            skeleton_graph_.RemoveEdge(i, j);
          }
        }
      }
    }

    if (cuda_use_managed_memory_) {
      std::swap(adjacency_matrix_, cuda_adjacency_matrix_);
    } else {
      cudaMemcpy(cuda_adjacency_matrix_, adjacency_matrix_, n_ * n_ * sizeof(int), cudaMemcpyHostToDevice);
    }
  }

  void TransferIntermediates(double* p_max_values, int* separation_sets, int max_level) {
    for (int i = 0; i < num_rows_; ++i) {
      p_max_values[i * n_ + i] = 1;
      for (int j = i + 1; j < n_; ++j) {
        if (!skeleton_graph_.HasEdge(i, j)) {
          p_max_values[i * n_ + j] = p_max_values_[i][j];
          p_max_values[j * n_ + i] = p_max_values_[i][j];
        } else {
          p_max_values[i * n_ + j] = -100000;
          p_max_values[j * n_ + i] = -100000;
        }
      }
    }

    for (int i = 0; i < num_rows_; ++i) {
      for (int j = i + 1; j < n_; ++j) {
        for (int k = 0; k < separation_sets_[i][j].size(); ++k) {
          separation_sets[(i * n_ + j) * max_level + k] = separation_sets_[i][j][k];
          separation_sets[(j * n_ + i) * max_level + k] = separation_sets_[i][j][k];
        }
      }
    }
  }

 private:
  bool cuda_use_managed_memory_;
  int* cuda_adjacency_matrix_;
  int* adjacency_matrix_;
  double* thresholds_;
  int num_rows_;
  int n_;
  GraphSkeleton skeleton_graph_;
  std::vector<std::vector<double>> p_max_values_;
  std::vector<std::vector<std::vector<int>>> separation_sets_;
};

void Skeleton(double* C, int* G, int* p, int* l, int* lMax, double* thresholds, double* pMax, int* sepSets,
              int* heterogeneousRows, int* managedMemory, char** outputFile) {
  // Bind arguments to local variables
  int& n = *p;
  int& level = *l;
  int& max_level = *lMax;
  int& num_heterogeneous_rows = *heterogeneousRows;
  bool use_managed_memory_cuda = *managedMemory > 0;
  char* output_file = outputFile[0];

  // Initialize benchmark data structures
  std::chrono::duration<double> gpu_initialization_duration;
  std::chrono::duration<double> cpu_initialization_duration;
  std::chrono::duration<double> gpu_termination_duration;
  std::chrono::duration<double> cpu_termination_duration;
  std::chrono::duration<double> gpu_transfer_duration;
  std::chrono::duration<double> cpu_transfer_duration;

  std::vector<std::chrono::duration<double>> gpu_level_durations(max_level + 1);
  std::vector<std::chrono::duration<double>> cpu_level_durations(max_level + 1);
  std::vector<std::chrono::duration<double>> merge_level_durations(max_level + 1);
  std::vector<std::chrono::duration<double>> heterogeneous_level_durations(max_level + 1);

  std::chrono::time_point<std::chrono::steady_clock> gpu_begin_time;
  std::chrono::time_point<std::chrono::steady_clock> cpu_begin_time;
  std::chrono::time_point<std::chrono::steady_clock> merge_begin_time;
  std::chrono::time_point<std::chrono::steady_clock> heterogeneous_begin_time;

  auto gpu_tic = [&]() { gpu_begin_time = std::chrono::steady_clock::now(); };
  auto cpu_tic = [&]() { cpu_begin_time = std::chrono::steady_clock::now(); };
  auto merge_tic = [&]() { merge_begin_time = std::chrono::steady_clock::now(); };
  auto heterogeneous_tic = [&]() { heterogeneous_begin_time = std::chrono::steady_clock::now(); };

  auto gpu_toc = [&]() { return std::chrono::steady_clock::now() - gpu_begin_time; };
  auto cpu_toc = [&]() { return std::chrono::steady_clock::now() - cpu_begin_time; };
  auto merge_toc = [&]() { return std::chrono::steady_clock::now() - merge_begin_time; };
  auto heterogeneous_toc = [&]() { return std::chrono::steady_clock::now() - heterogeneous_begin_time; };

  // Initialize GPU data structures
  gpu_tic();

  double* C_cuda;
  double* pMax_cuda;
  int* G_cuda;
  int* nprime_cuda;
  int* SepSet_cuda;
  int* GPrime_cuda;
  int* mutex_cuda;

  int nprime = 0;
  dim3 BLOCKS_PER_GRID;
  dim3 THREADS_PER_BLOCK;

  bool FinishFlag = false;

  cudaMalloc((void**)&mutex_cuda, n * n * sizeof(int));
  cudaMalloc((void**)&nprime_cuda, 1 * sizeof(int));
  cudaMalloc((void**)&GPrime_cuda, n * n * sizeof(int));
  cudaMalloc((void**)&C_cuda, n * n * sizeof(double));

  if (use_managed_memory_cuda) {
    cudaMallocManaged((void**)&G_cuda, n * n * sizeof(int));
  } else {
    cudaMalloc((void**)&G_cuda, n * n * sizeof(int));
  }

  cudaMalloc((void**)&pMax_cuda, n * n * sizeof(double));
  cudaMalloc((void**)&SepSet_cuda, n * n * ML * sizeof(int));

  cudaMemcpy(C_cuda, C, n * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemset(mutex_cuda, 0, n * n * sizeof(int));

  cudaDeviceSynchronize();

  gpu_initialization_duration = gpu_toc();

  // Initialize CPU data structures
  cpu_tic();

  HeterogeneousExecutionManager* heterogeneous_execution_manager =
      new HeterogeneousExecutionManager(use_managed_memory_cuda, G_cuda, G, C, thresholds, num_heterogeneous_rows, n);

  auto calculate_heterogeneous_level = [&](int level) {
    cpu_tic();
    heterogeneous_execution_manager->CalculateLevel(level);
    cpu_level_durations[level] = cpu_toc();
  };

  cpu_initialization_duration = cpu_toc();

  //----------------------------------------------------------

  for (level = 0; level <= ML && !FinishFlag && level <= max_level; ++level) {
    if (level == 0) {
      heterogeneous_tic();

      std::thread t(calculate_heterogeneous_level, level);

      gpu_tic();
      BLOCKS_PER_GRID = dim3(1, 1, 1);
      THREADS_PER_BLOCK = dim3(32, 32, 1);
      if ((n * n) >= 1024) {
        BLOCKS_PER_GRID = dim3(ceil(((double)(n)) / 32.0), ceil(((double)(n)) / 32.0), 1);
        THREADS_PER_BLOCK = dim3(32, 32, 1);
      }
      cal_Indepl0<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(C_cuda, G_cuda, thresholds[0], pMax_cuda, n,
                                                          num_heterogeneous_rows);
      cudaDeviceSynchronize();

      BLOCKS_PER_GRID = dim3(n * n, 1, 1);
      THREADS_PER_BLOCK = dim3(ML, 1, 1);
      SepSet_initialize<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(SepSet_cuda, n);
      cudaDeviceSynchronize();
      gpu_level_durations[level] = gpu_toc();

      t.join();

      merge_tic();
      heterogeneous_execution_manager->MergeLevel(level);
      merge_level_durations[level] = merge_toc();

      heterogeneous_level_durations[level] = heterogeneous_toc();
    } else {
      //================================> Start Scan Process <===============================
      cudaMemset(nprime_cuda, 0, 1 * sizeof(int));
      BLOCKS_PER_GRID = dim3(1, n, 1);
      THREADS_PER_BLOCK = dim3(1024, 1, 1);
      scan_compact<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, n * sizeof(int)>>>(GPrime_cuda, G_cuda, n, nprime_cuda);
      cudaDeviceSynchronize();
      cudaMemcpy(&nprime, nprime_cuda, 1 * sizeof(int), cudaMemcpyDeviceToHost);

      //================================> Begin The Gaussian CI Test  <==============================
      // CHeck whether a CI test is possible
      if (nprime - 1 < level) {
        --level;
        FinishFlag = true;
        break;
      }

      heterogeneous_tic();

      std::thread t(calculate_heterogeneous_level, level);

      gpu_tic();
      if (level == 1) {
        BLOCKS_PER_GRID = dim3(NumOfBlockForEachNodeL1, n, 1);
        THREADS_PER_BLOCK = dim3(ParGivenL1, 1, 1);
        cal_Indepl1<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, nprime * sizeof(int)>>>(
            C_cuda, G_cuda, GPrime_cuda, mutex_cuda, SepSet_cuda, pMax_cuda, thresholds[1], n, num_heterogeneous_rows);

        cudaDeviceSynchronize();
      } else if (level == 2) {
        BLOCKS_PER_GRID = dim3(NumOfBlockForEachNodeL2, n, 1);
        THREADS_PER_BLOCK = dim3(ParGivenL2, 1, 1);
        cal_Indepl2<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, nprime * sizeof(int)>>>(
            C_cuda, G_cuda, GPrime_cuda, mutex_cuda, SepSet_cuda, pMax_cuda, thresholds[2], n, num_heterogeneous_rows);

        cudaDeviceSynchronize();
      } else if (level == 3) {
        BLOCKS_PER_GRID = dim3(NumOfBlockForEachNodeL3, n, 1);
        THREADS_PER_BLOCK = dim3(ParGivenL3, 1, 1);
        cal_Indepl3<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, nprime * sizeof(int)>>>(
            C_cuda, G_cuda, GPrime_cuda, mutex_cuda, SepSet_cuda, pMax_cuda, thresholds[3], n, num_heterogeneous_rows);

        cudaDeviceSynchronize();
      } else if (level == 4) {
        BLOCKS_PER_GRID = dim3(NumOfBlockForEachNodeL4, n, 1);
        THREADS_PER_BLOCK = dim3(ParGivenL4, 1, 1);
        cal_Indepl4<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, nprime * sizeof(int)>>>(
            C_cuda, G_cuda, GPrime_cuda, mutex_cuda, SepSet_cuda, pMax_cuda, thresholds[4], n, num_heterogeneous_rows);

        cudaDeviceSynchronize();
      } else if (level == 5) {
        BLOCKS_PER_GRID = dim3(NumOfBlockForEachNodeL5, n, 1);
        THREADS_PER_BLOCK = dim3(ParGivenL5, 1, 1);
        cal_Indepl5<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, nprime * sizeof(int)>>>(
            C_cuda, G_cuda, GPrime_cuda, mutex_cuda, SepSet_cuda, pMax_cuda, thresholds[5], n, num_heterogeneous_rows);

        cudaDeviceSynchronize();
      } else if (level == 6) {
        BLOCKS_PER_GRID = dim3(NumOfBlockForEachNodeL6, n, 1);
        THREADS_PER_BLOCK = dim3(ParGivenL6, 1, 1);
        cal_Indepl6<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, nprime * sizeof(int)>>>(
            C_cuda, G_cuda, GPrime_cuda, mutex_cuda, SepSet_cuda, pMax_cuda, thresholds[6], n, num_heterogeneous_rows);

        cudaDeviceSynchronize();
      } else if (level == 7) {
        BLOCKS_PER_GRID = dim3(NumOfBlockForEachNodeL7, n, 1);
        THREADS_PER_BLOCK = dim3(ParGivenL7, 1, 1);
        cal_Indepl7<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, nprime * sizeof(int)>>>(
            C_cuda, G_cuda, GPrime_cuda, mutex_cuda, SepSet_cuda, pMax_cuda, thresholds[7], n, num_heterogeneous_rows);

        cudaDeviceSynchronize();
      } else if (level == 8) {
        BLOCKS_PER_GRID = dim3(NumOfBlockForEachNodeL8, n, 1);
        THREADS_PER_BLOCK = dim3(ParGivenL8, 1, 1);
        cal_Indepl8<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, nprime * sizeof(int)>>>(
            C_cuda, G_cuda, GPrime_cuda, mutex_cuda, SepSet_cuda, pMax_cuda, thresholds[8], n, num_heterogeneous_rows);

        cudaDeviceSynchronize();
      } else if (level == 9) {
        BLOCKS_PER_GRID = dim3(NumOfBlockForEachNodeL9, n, 1);
        THREADS_PER_BLOCK = dim3(ParGivenL9, 1, 1);
        cal_Indepl9<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, nprime * sizeof(int)>>>(
            C_cuda, G_cuda, GPrime_cuda, mutex_cuda, SepSet_cuda, pMax_cuda, thresholds[9], n, num_heterogeneous_rows);

        cudaDeviceSynchronize();
      } else if (level == 10) {
        BLOCKS_PER_GRID = dim3(NumOfBlockForEachNodeL10, n, 1);
        THREADS_PER_BLOCK = dim3(ParGivenL10, 1, 1);
        cal_Indepl10<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, nprime * sizeof(int)>>>(
            C_cuda, G_cuda, GPrime_cuda, mutex_cuda, SepSet_cuda, pMax_cuda, thresholds[10], n, num_heterogeneous_rows);

        cudaDeviceSynchronize();
      } else if (level == 11) {
        BLOCKS_PER_GRID = dim3(NumOfBlockForEachNodeL11, n, 1);
        THREADS_PER_BLOCK = dim3(ParGivenL11, 1, 1);
        cal_Indepl11<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, nprime * sizeof(int)>>>(
            C_cuda, G_cuda, GPrime_cuda, mutex_cuda, SepSet_cuda, pMax_cuda, thresholds[11], n, num_heterogeneous_rows);

        cudaDeviceSynchronize();
      } else if (level == 12) {
        BLOCKS_PER_GRID = dim3(NumOfBlockForEachNodeL12, n, 1);
        THREADS_PER_BLOCK = dim3(ParGivenL12, 1, 1);
        cal_Indepl12<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, nprime * sizeof(int)>>>(
            C_cuda, G_cuda, GPrime_cuda, mutex_cuda, SepSet_cuda, pMax_cuda, thresholds[12], n, num_heterogeneous_rows);

        cudaDeviceSynchronize();
      } else if (level == 13) {
        BLOCKS_PER_GRID = dim3(NumOfBlockForEachNodeL13, n, 1);
        THREADS_PER_BLOCK = dim3(ParGivenL13, 1, 1);
        cal_Indepl13<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, nprime * sizeof(int)>>>(
            C_cuda, G_cuda, GPrime_cuda, mutex_cuda, SepSet_cuda, pMax_cuda, thresholds[13], n, num_heterogeneous_rows);

        cudaDeviceSynchronize();
      } else if (level == 14) {
        BLOCKS_PER_GRID = dim3(NumOfBlockForEachNodeL14, n, 1);
        THREADS_PER_BLOCK = dim3(ParGivenL14, 1, 1);
        cal_Indepl14<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, nprime * sizeof(int)>>>(
            C_cuda, G_cuda, GPrime_cuda, mutex_cuda, SepSet_cuda, pMax_cuda, thresholds[14], n, num_heterogeneous_rows);

        cudaDeviceSynchronize();
      } else {
        // TODO: level > 14
      }
      gpu_level_durations[level] = gpu_toc();

      t.join();

      merge_tic();
      heterogeneous_execution_manager->MergeLevel(level);
      merge_level_durations[level] = merge_toc();

      heterogeneous_level_durations[level] = heterogeneous_toc();
    }
  }

  // Transfer GPU data structures back
  gpu_tic();

  cudaMemcpy(G, G_cuda, n * n * sizeof(int), cudaMemcpyDeviceToHost);

  cudaMemcpy(sepSets + (num_heterogeneous_rows * n * ML), SepSet_cuda + (num_heterogeneous_rows * n * ML),
             (n - num_heterogeneous_rows) * n * ML * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(pMax + (num_heterogeneous_rows * n), pMax_cuda + (num_heterogeneous_rows * n),
             (n - num_heterogeneous_rows) * n * sizeof(double), cudaMemcpyDeviceToHost);

  for (int i = num_heterogeneous_rows; i < n; i++) {
    pMax[i * n + i] = 1;
    for (int j = (i + 1); j < n; j++) {
      if (G[i * n + j] == 0) {
        const double t = fmax(pMax[j * n + i], pMax[i * n + j]);
        pMax[j * n + i] = t;
        pMax[i * n + j] = t;
      } else {
        pMax[j * n + i] = -100000;
        pMax[i * n + j] = -100000;
      }
    }
  }

  gpu_transfer_duration = gpu_toc();

  // Transfer CPU data structures back
  cpu_tic();

  heterogeneous_execution_manager->TransferIntermediates(pMax, sepSets, max_level);

  cpu_transfer_duration = cpu_toc();

  // Terminate GPU resources
  gpu_tic();

  cudaFree(SepSet_cuda);
  cudaFree(C_cuda);
  cudaFree(GPrime_cuda);
  cudaFree(G_cuda);
  cudaFree(mutex_cuda);
  cudaFree(pMax_cuda);

  cudaDeviceSynchronize();

  gpu_termination_duration = gpu_toc();

  // Terminate CPU resources
  cpu_tic();

  delete heterogeneous_execution_manager;

  cpu_termination_duration = cpu_toc();

  // Write benchmark results
  std::chrono::duration<double> heterogeneous_total_duration;
  std::stringstream result_stream;

  result_stream << std::fixed << std::setprecision(9);

  heterogeneous_total_duration += gpu_initialization_duration + cpu_initialization_duration;
  result_stream << gpu_initialization_duration.count() << "," << cpu_initialization_duration.count() << ",";

  for (int i = 0; i <= max_level; ++i) {
    heterogeneous_total_duration += heterogeneous_level_durations[i];
    result_stream << gpu_level_durations[i].count() << "," << cpu_level_durations[i].count() << ","
                  << merge_level_durations[i].count() << "," << heterogeneous_level_durations[i].count() << ",";
  }

  heterogeneous_total_duration += gpu_transfer_duration + cpu_transfer_duration;
  result_stream << gpu_transfer_duration.count() << "," << cpu_transfer_duration.count() << ",";

  heterogeneous_total_duration += gpu_termination_duration + cpu_termination_duration;
  result_stream << gpu_termination_duration.count() << "," << cpu_termination_duration.count() << ",";

  result_stream << heterogeneous_total_duration.count() << "\n";

  std::ofstream output_stream(output_file, std::ofstream::app);

  if (output_stream) {
    output_stream << result_stream.str();

    output_stream.close();
  } else {
    std::cout << result_stream.str();
  }
}  // Skeleton

__global__ void SepSet_initialize(int* SepSet, int size) {
  int row = blockIdx.x;
  SepSet[row * ML + threadIdx.x] = -1;
}

__global__ void cal_Indepl0(double* C, int* G, double th, double* pMax, int n, int num_heterogeneous_rows) {
  int row = num_heterogeneous_rows + blockDim.x * blockIdx.x + threadIdx.x;
  int col = num_heterogeneous_rows + blockDim.y * blockIdx.y + threadIdx.y;
  if (row < col && col < n) {
    double res = C[row * n + col];
    res = abs(0.5 * log(abs((1 + res) / (1 - res))));
    if (res < th) {
      pMax[row * n + col] = res;
      G[row * n + col] = 0;
      G[col * n + row] = 0;
    } else {
      G[row * n + col] = 1;
      G[col * n + row] = 1;
    }
  }
  if (row == col && col < n) {
    G[row * n + col] = 0;
    G[col * n + row] = 0;
  }
}

__global__ void cal_Indepl1(double* C, int* G, int* GPrime, int* mutex, int* Sepset, double* pMax, double th, int n,
                            int num_heterogeneous_rows) {
  int YIdx;
  int XIdx = blockIdx.y;
  int NbrIdxPointer;
  int NbrIdx;
  int SizeOfArr;
  int NumberOfJump;
  int NumOfGivenJump;
  __shared__ int NoEdgeFlag;
  double M0;
  double H[2][2];
  double M1[2];
  double rho, Z;
  extern __shared__ int G_Chunk[];

  NoEdgeFlag = 0;
  SizeOfArr = GPrime[XIdx * n + n - 1];
  if ((SizeOfArr % ParGivenL1) == 0) {
    NumberOfJump = SizeOfArr / ParGivenL1;
  } else {
    NumberOfJump = SizeOfArr / ParGivenL1 + 1;
  }
  // Copy Row Xid from GPrime to G_chunck
  for (int cnt = 0; cnt < NumberOfJump; cnt++) {
    if ((threadIdx.x + cnt * ParGivenL1) < SizeOfArr) {
      G_Chunk[threadIdx.x + cnt * ParGivenL1] = GPrime[XIdx * n + threadIdx.x + cnt * ParGivenL1];
    }
    __syncthreads();
  }

  if ((SizeOfArr % (ParGivenL1 * NumOfBlockForEachNodeL1)) == 0) {
    NumOfGivenJump = SizeOfArr / (ParGivenL1 * NumOfBlockForEachNodeL1);
  } else {
    NumOfGivenJump = SizeOfArr / (ParGivenL1 * NumOfBlockForEachNodeL1) + 1;
  }

  for (int d1 = 0; d1 < NumOfGivenJump; d1++) {
    __syncthreads();
    if (NoEdgeFlag == 1) {
      return;
    }
    __syncthreads();
    NbrIdxPointer = threadIdx.x + blockIdx.x * ParGivenL1 + d1 * ParGivenL1 * NumOfBlockForEachNodeL1;
    NoEdgeFlag = 1;
    __syncthreads();
    if (NbrIdxPointer < SizeOfArr) {
      NbrIdx = G_Chunk[NbrIdxPointer];
      M1[0] = C[XIdx * n + NbrIdx];
      for (int d2 = 0; d2 < SizeOfArr; d2++) {
        if (d2 == NbrIdxPointer) {
          continue;
        }
        YIdx = G_Chunk[d2];

        if (XIdx < num_heterogeneous_rows || YIdx < num_heterogeneous_rows) {
          continue;
        }

        if (G[XIdx * n + YIdx] == 1) {
          NoEdgeFlag = 0;
          M0 = C[XIdx * n + YIdx];
          M1[1] = C[YIdx * n + NbrIdx];

          H[0][0] = 1 - (M1[0] * M1[0]);
          H[0][1] = M0 - (M1[0] * M1[1]);
          H[1][1] = 1 - (M1[1] * M1[1]);

          rho = H[0][1] / (sqrt(fabs(H[0][0])) * sqrt(fabs(H[1][1])));
          Z = fabs(0.5 * (log(fabs((1 + rho))) - log(fabs(1 - rho))));

          if (Z < th) {
            if (atomicCAS(&mutex[XIdx * n + YIdx], 0, 1) == 0) {
              G[XIdx * n + YIdx] = 0;
              G[YIdx * n + XIdx] = 0;
              pMax[XIdx * n + YIdx] = Z;
              Sepset[(XIdx * n + YIdx) * ML] = NbrIdx;
            }
          }
        }
      }
    }
  }
}

__global__ void cal_Indepl2(double* C, int* G, int* GPrime, int* mutex, int* Sepset, double* pMax, double th, int n,
                            int num_heterogeneous_rows) {
  int YIdx;
  int XIdx = blockIdx.y;
  int NbrIdxPointer[2];
  int NbrIdx[2];
  int SizeOfArr;
  int NumberOfJump;
  int NumOfGivenJump;
  int NumOfComb;
  __shared__ int NoEdgeFlag;
  double M0;
  double M1[2][2];
  double M2[2][2];
  double M2Inv[2][2];
  double M1MulM2Inv[2][2];
  double H[2][2];
  double rho;
  double Z;
  // Lock WriteSepSetLock;

  extern __shared__ int G_Chunk[];

  NoEdgeFlag = 0;
  SizeOfArr = GPrime[XIdx * n + n - 1];
  if (SizeOfArr <= 2) {
    return;
  }

  if ((SizeOfArr % ParGivenL2) == 0) {
    NumberOfJump = SizeOfArr / ParGivenL2;
  } else {
    NumberOfJump = SizeOfArr / ParGivenL2 + 1;
  }
  // Copy Row Xid from GPrime to G_chunck
  for (int cnt = 0; cnt < NumberOfJump; cnt++) {
    if ((threadIdx.x + cnt * ParGivenL2) < SizeOfArr) {
      G_Chunk[threadIdx.x + cnt * ParGivenL2] = GPrime[XIdx * n + threadIdx.x + cnt * ParGivenL2];
    }
    __syncthreads();
  }

  BINOM(SizeOfArr, 2, &NumOfComb);
  if ((NumOfComb % (ParGivenL2 * NumOfBlockForEachNodeL2)) == 0) {
    NumOfGivenJump = NumOfComb / (ParGivenL2 * NumOfBlockForEachNodeL2);
  } else {
    NumOfGivenJump = NumOfComb / (ParGivenL2 * NumOfBlockForEachNodeL2) + 1;
  }

  for (int d1 = 0; d1 < NumOfGivenJump; d1++) {
    __syncthreads();
    if (NoEdgeFlag == 1) {
      return;
    }
    if ((threadIdx.x + blockIdx.x * ParGivenL2 + d1 * ParGivenL2 * NumOfBlockForEachNodeL2) < NumOfComb) {
      __syncthreads();
      NoEdgeFlag = 1;
      __syncthreads();
      IthCombination(NbrIdxPointer, SizeOfArr, 2,
                     threadIdx.x + blockIdx.x * ParGivenL2 + d1 * ParGivenL2 * NumOfBlockForEachNodeL2 + 1);
      NbrIdx[0] = G_Chunk[NbrIdxPointer[0] - 1];
      NbrIdx[1] = G_Chunk[NbrIdxPointer[1] - 1];
      M2[0][1] = C[NbrIdx[0] * n + NbrIdx[1]];
      M2[1][0] = M2[0][1];
      M2[1][1] = 1;
      M2[0][0] = 1;

      M1[0][1] = C[XIdx * n + NbrIdx[1]];
      M1[0][0] = C[XIdx * n + NbrIdx[0]];
      pseudoinversel2(M2, M2Inv);
      for (int d2 = 0; d2 < SizeOfArr; d2++) {
        if ((d2 == (NbrIdxPointer[0] - 1)) || (d2 == (NbrIdxPointer[1] - 1))) {
          continue;
        }
        YIdx = G_Chunk[d2];

        if (XIdx < num_heterogeneous_rows || YIdx < num_heterogeneous_rows) {
          continue;
        }

        if (G[XIdx * n + YIdx] == 1) {
          NoEdgeFlag = 0;
          M0 = C[XIdx * n + YIdx];
          M1[1][0] = C[YIdx * n + NbrIdx[0]];
          M1[1][1] = C[YIdx * n + NbrIdx[1]];
          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 2; c2++) {
              M1MulM2Inv[c1][c2] = 0;
              for (int c3 = 0; c3 < 2; c3++) M1MulM2Inv[c1][c2] += M1[c1][c3] * M2Inv[c3][c2];
            }
          }
          H[0][0] = 1 - (M1MulM2Inv[0][0] * M1[0][0] + M1MulM2Inv[0][1] * M1[0][1]);
          H[0][1] = M0 - (M1MulM2Inv[0][0] * M1[1][0] + M1MulM2Inv[0][1] * M1[1][1]);
          H[1][1] = 1 - (M1MulM2Inv[1][0] * M1[1][0] + M1MulM2Inv[1][1] * M1[1][1]);

          rho = H[0][1] / (sqrt(abs(H[0][0] * H[1][1])));
          Z = 0.5 * abs(log(abs((1 + rho) / (1 - rho))));

          if (Z < th) {
            if (atomicCAS(&mutex[XIdx * n + YIdx], 0, 1) == 0) {  // lock
              G[XIdx * n + YIdx] = 0;
              G[YIdx * n + XIdx] = 0;
              pMax[XIdx * n + YIdx] = Z;
              Sepset[(XIdx * n + YIdx) * ML] = NbrIdx[0];
              Sepset[(XIdx * n + YIdx) * ML + 1] = NbrIdx[1];
            }
          }
        }
      }
    }
  }
}

__global__ void cal_Indepl3(double* C, int* G, int* GPrime, int* mutex, int* Sepset, double* pMax, double th, int n,
                            int num_heterogeneous_rows) {
  int YIdx;
  int XIdx = blockIdx.y;
  int NbrIdxPointer[3];
  int SizeOfArr;
  int NumberOfJump;
  int NumOfGivenJump;
  int NumOfComb;
  __shared__ int NoEdgeFlag;
  int NbrIdx[3];
  double M0;
  double M1[2][3];
  double M2[3][3];
  double M2Inv[3][3];
  double M1MulM2Inv[2][3];
  double H[2][2];
  double rho;
  double Z;
  // Lock WriteSepSetLock;
  extern __shared__ int G_Chunk[];
  NoEdgeFlag = 0;
  SizeOfArr = GPrime[XIdx * n + n - 1];
  if (SizeOfArr <= 3) {
    return;
  }

  if ((SizeOfArr % ParGivenL3) == 0) {
    NumberOfJump = SizeOfArr / ParGivenL3;
  } else {
    NumberOfJump = SizeOfArr / ParGivenL3 + 1;
  }
  // Copy Row Xid from GPrime to G_chunck
  for (int cnt = 0; cnt < NumberOfJump; cnt++) {
    if ((threadIdx.x + cnt * ParGivenL3) < SizeOfArr) {
      G_Chunk[threadIdx.x + cnt * ParGivenL3] = GPrime[XIdx * n + threadIdx.x + cnt * ParGivenL3];
    }
    __syncthreads();
  }

  BINOM(SizeOfArr, 3, &NumOfComb);
  if ((NumOfComb % (ParGivenL3 * NumOfBlockForEachNodeL3)) == 0) {
    NumOfGivenJump = NumOfComb / (ParGivenL3 * NumOfBlockForEachNodeL3);
  } else {
    NumOfGivenJump = NumOfComb / (ParGivenL3 * NumOfBlockForEachNodeL3) + 1;
  }
  __syncthreads();
  for (int d1 = 0; d1 < NumOfGivenJump; d1++) {
    __syncthreads();
    if (NoEdgeFlag == 1) {
      return;
    }
    if (threadIdx.x + blockIdx.x * ParGivenL3 + d1 * ParGivenL3 * NumOfBlockForEachNodeL3 < NumOfComb) {
      __syncthreads();
      NoEdgeFlag = 1;
      __syncthreads();
      IthCombination(NbrIdxPointer, SizeOfArr, 3,
                     threadIdx.x + blockIdx.x * ParGivenL3 + d1 * ParGivenL3 * NumOfBlockForEachNodeL3 + 1);
      NbrIdx[0] = G_Chunk[NbrIdxPointer[0] - 1];
      NbrIdx[1] = G_Chunk[NbrIdxPointer[1] - 1];
      NbrIdx[2] = G_Chunk[NbrIdxPointer[2] - 1];
      M2[0][0] = 1;
      M2[0][1] = C[NbrIdx[0] * n + NbrIdx[1]];
      M2[0][2] = C[NbrIdx[0] * n + NbrIdx[2]];
      M2[1][0] = M2[0][1];
      M2[1][1] = 1;
      M2[1][2] = C[NbrIdx[1] * n + NbrIdx[2]];
      M2[2][0] = M2[0][2];
      M2[2][1] = M2[1][2];
      M2[2][2] = 1;

      M1[0][0] = C[XIdx * n + NbrIdx[0]];
      M1[0][1] = C[XIdx * n + NbrIdx[1]];
      M1[0][2] = C[XIdx * n + NbrIdx[2]];

      pseudoinversel3(M2, M2Inv);
      for (int d2 = 0; d2 < SizeOfArr; d2++) {
        if ((d2 == (NbrIdxPointer[0] - 1)) || (d2 == (NbrIdxPointer[1] - 1)) || (d2 == (NbrIdxPointer[2] - 1))) {
          continue;
        }
        YIdx = G_Chunk[d2];

        if (XIdx < num_heterogeneous_rows || YIdx < num_heterogeneous_rows) {
          continue;
        }

        if (G[XIdx * n + YIdx] == 1) {
          NoEdgeFlag = 0;
          M0 = C[XIdx * n + YIdx];

          M1[1][0] = C[YIdx * n + NbrIdx[0]];
          M1[1][1] = C[YIdx * n + NbrIdx[1]];
          M1[1][2] = C[YIdx * n + NbrIdx[2]];
          // Begin to calculate I2Inv Using pseudo-inverse
          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 3; c2++) {
              M1MulM2Inv[c1][c2] = 0;
              for (int c3 = 0; c3 < 3; c3++) M1MulM2Inv[c1][c2] += M1[c1][c3] * M2Inv[c3][c2];
            }
          }

          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 2; c2++) {
              H[c1][c2] = 0;
              for (int c3 = 0; c3 < 3; c3++) H[c1][c2] += M1MulM2Inv[c1][c3] * M1[c2][c3];
            }
          }
          H[0][0] = 1 - H[0][0];
          H[0][1] = M0 - H[0][1];
          H[1][1] = 1 - H[1][1];

          rho = H[0][1] / (sqrt(abs(H[0][0] * H[1][1])));
          Z = abs(0.5 * log(abs((1 + rho) / (1 - rho))));

          if (Z < th) {
            if (atomicCAS(&mutex[XIdx * n + YIdx], 0, 1) == 0) {  // lock
              G[XIdx * n + YIdx] = 0;
              G[YIdx * n + XIdx] = 0;
              pMax[XIdx * n + YIdx] = Z;
              Sepset[(XIdx * n + YIdx) * ML] = NbrIdx[0];
              Sepset[(XIdx * n + YIdx) * ML + 1] = NbrIdx[1];
              Sepset[(XIdx * n + YIdx) * ML + 2] = NbrIdx[2];
            }
          }
        }
      }
    }
  }
}

__global__ void cal_Indepl4(double* C, int* G, int* GPrime, int* mutex, int* Sepset, double* pMax, double th, int n,
                            int num_heterogeneous_rows) {
  int YIdx;
  int XIdx = blockIdx.y;
  int NbrIdxPointer[4];
  int SizeOfArr;
  int NumberOfJump;
  int NumOfGivenJump;
  int NumOfComb;
  __shared__ int NoEdgeFlag;
  int NbrIdx[4];

  double M0;
  double M1[2][4];
  double M2[4][4];
  double M2Inv[4][4];
  double M1MulM2Inv[2][4];
  double H[2][2];
  double rho;
  double Z;

  double v[4][4];
  double w[4], rv1[4];
  double res1[4][4];
  // Lock WriteSepSetLock;
  extern __shared__ int G_Chunk[];

  NoEdgeFlag = 0;
  SizeOfArr = GPrime[XIdx * n + n - 1];
  if (SizeOfArr <= 4) {
    return;
  }
  if ((SizeOfArr % ParGivenL4) == 0) {
    NumberOfJump = SizeOfArr / ParGivenL4;
  } else {
    NumberOfJump = SizeOfArr / ParGivenL4 + 1;
  }
  // Copy Row Xid from GPrime to G_chunck
  for (int cnt = 0; cnt < NumberOfJump; cnt++) {
    if ((threadIdx.x + cnt * ParGivenL4) < SizeOfArr) {
      G_Chunk[threadIdx.x + cnt * ParGivenL4] = GPrime[XIdx * n + threadIdx.x + cnt * ParGivenL4];
    }
    __syncthreads();
  }

  BINOM(SizeOfArr, 4, &NumOfComb);
  if ((NumOfComb % (ParGivenL4 * NumOfBlockForEachNodeL4)) == 0) {
    NumOfGivenJump = NumOfComb / (ParGivenL4 * NumOfBlockForEachNodeL4);
  } else {
    NumOfGivenJump = NumOfComb / (ParGivenL4 * NumOfBlockForEachNodeL4) + 1;
  }
  __syncthreads();
  for (int d1 = 0; d1 < NumOfGivenJump; d1++) {
    __syncthreads();
    if (NoEdgeFlag == 1) {
      return;
    }
    if (threadIdx.x + blockIdx.x * ParGivenL4 + d1 * ParGivenL4 * NumOfBlockForEachNodeL4 < NumOfComb) {
      __syncthreads();
      NoEdgeFlag = 1;
      __syncthreads();
      IthCombination(NbrIdxPointer, SizeOfArr, 4,
                     threadIdx.x + blockIdx.x * ParGivenL4 + d1 * ParGivenL4 * NumOfBlockForEachNodeL4 + 1);
      NbrIdx[0] = G_Chunk[NbrIdxPointer[0] - 1];
      NbrIdx[1] = G_Chunk[NbrIdxPointer[1] - 1];
      NbrIdx[2] = G_Chunk[NbrIdxPointer[2] - 1];
      NbrIdx[3] = G_Chunk[NbrIdxPointer[3] - 1];

      M2[0][0] = 1;
      M2[0][1] = C[NbrIdx[0] * n + NbrIdx[1]];
      M2[0][2] = C[NbrIdx[0] * n + NbrIdx[2]];
      M2[0][3] = C[NbrIdx[0] * n + NbrIdx[3]];

      M2[1][0] = M2[0][1];
      M2[1][1] = 1;
      M2[1][2] = C[NbrIdx[1] * n + NbrIdx[2]];
      M2[1][3] = C[NbrIdx[1] * n + NbrIdx[3]];

      M2[2][0] = M2[0][2];
      M2[2][1] = M2[1][2];
      M2[2][2] = 1;
      M2[2][3] = C[NbrIdx[2] * n + NbrIdx[3]];

      M2[3][0] = M2[0][3];
      M2[3][1] = M2[1][3];
      M2[3][2] = M2[2][3];
      M2[3][3] = 1;

      M1[0][0] = C[XIdx * n + NbrIdx[0]];
      M1[0][1] = C[XIdx * n + NbrIdx[1]];
      M1[0][2] = C[XIdx * n + NbrIdx[2]];
      M1[0][3] = C[XIdx * n + NbrIdx[3]];
      pseudoinversel4(M2, M2Inv, v, rv1, w, res1);
      for (int d2 = 0; d2 < SizeOfArr; d2++) {
        if ((d2 == (NbrIdxPointer[0] - 1)) || (d2 == (NbrIdxPointer[1] - 1)) || (d2 == (NbrIdxPointer[2] - 1)) ||
            (d2 == (NbrIdxPointer[3] - 1))) {
          continue;
        }
        YIdx = G_Chunk[d2];

        if (XIdx < num_heterogeneous_rows || YIdx < num_heterogeneous_rows) {
          continue;
        }

        if (G[XIdx * n + YIdx] == 1) {
          NoEdgeFlag = 0;
          M0 = C[XIdx * n + YIdx];

          M1[1][0] = C[YIdx * n + NbrIdx[0]];
          M1[1][1] = C[YIdx * n + NbrIdx[1]];
          M1[1][2] = C[YIdx * n + NbrIdx[2]];
          M1[1][3] = C[YIdx * n + NbrIdx[3]];
          // Begin to calculate I2Inv Using pseudo-inverse
          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 4; c2++) {
              M1MulM2Inv[c1][c2] = 0;
              for (int c3 = 0; c3 < 4; c3++) M1MulM2Inv[c1][c2] += M1[c1][c3] * M2Inv[c3][c2];
            }
          }

          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 2; c2++) {
              H[c1][c2] = 0;
              for (int c3 = 0; c3 < 4; c3++) H[c1][c2] += M1MulM2Inv[c1][c3] * M1[c2][c3];
            }
          }
          H[0][0] = 1 - H[0][0];
          H[0][1] = M0 - H[0][1];
          H[1][1] = 1 - H[1][1];

          rho = H[0][1] / (sqrt(abs(H[0][0] * H[1][1])));
          Z = abs(0.5 * log(abs((1 + rho) / (1 - rho))));

          if (Z < th) {
            if (atomicCAS(&mutex[XIdx * n + YIdx], 0, 1) == 0) {  // lock
              G[XIdx * n + YIdx] = 0;
              G[YIdx * n + XIdx] = 0;
              pMax[XIdx * n + YIdx] = Z;
              Sepset[(XIdx * n + YIdx) * ML] = NbrIdx[0];
              Sepset[(XIdx * n + YIdx) * ML + 1] = NbrIdx[1];
              Sepset[(XIdx * n + YIdx) * ML + 2] = NbrIdx[2];
              Sepset[(XIdx * n + YIdx) * ML + 3] = NbrIdx[3];
            }
          }
        }
      }
    }
  }
}

__global__ void cal_Indepl5(double* C, int* G, int* GPrime, int* mutex, int* Sepset, double* pMax, double th, int n,
                            int num_heterogeneous_rows) {
  int YIdx;
  int XIdx = blockIdx.y;
  int NbrIdxPointer[5];
  int SizeOfArr;
  int NumberOfJump;
  int NumOfGivenJump;
  int NumOfComb;
  __shared__ int NoEdgeFlag;
  int NbrIdx[5];

  double M0;
  double M1[2][5];
  double M2[5][5];
  double M2Inv[5][5];
  double M1MulM2Inv[2][5];
  double H[2][2];
  double rho;
  double Z;
  extern __shared__ int G_Chunk[];
  // pseudo-inverse parameter
  double v[5][5];
  double w[5], rv1[5];
  double res1[5][5];
  NoEdgeFlag = 0;
  SizeOfArr = GPrime[XIdx * n + n - 1];
  if (SizeOfArr <= 5) {
    return;
  }

  if ((SizeOfArr % ParGivenL5) == 0) {
    NumberOfJump = SizeOfArr / ParGivenL5;
  } else {
    NumberOfJump = SizeOfArr / ParGivenL5 + 1;
  }
  // Copy Row Xid from GPrime to G_chunck
  for (int cnt = 0; cnt < NumberOfJump; cnt++) {
    if ((threadIdx.x + cnt * ParGivenL5) < SizeOfArr) {
      G_Chunk[threadIdx.x + cnt * ParGivenL5] = GPrime[XIdx * n + threadIdx.x + cnt * ParGivenL5];
    }
    __syncthreads();
  }

  BINOM(SizeOfArr, 5, &NumOfComb);
  if ((NumOfComb % (ParGivenL5 * NumOfBlockForEachNodeL5)) == 0) {
    NumOfGivenJump = NumOfComb / (ParGivenL5 * NumOfBlockForEachNodeL5);
  } else {
    NumOfGivenJump = NumOfComb / (ParGivenL5 * NumOfBlockForEachNodeL5) + 1;
  }

  for (int d1 = 0; d1 < NumOfGivenJump; d1++) {
    __syncthreads();
    if (NoEdgeFlag == 1) {
      return;
    }
    if (threadIdx.x + blockIdx.x * ParGivenL5 + d1 * ParGivenL5 * NumOfBlockForEachNodeL5 < NumOfComb) {
      __syncthreads();
      NoEdgeFlag = 1;
      __syncthreads();
      IthCombination(NbrIdxPointer, SizeOfArr, 5,
                     threadIdx.x + blockIdx.x * ParGivenL5 + d1 * ParGivenL5 * NumOfBlockForEachNodeL5 + 1);
      for (int tmp = 0; tmp < 5; tmp++) {
        NbrIdx[tmp] = G_Chunk[NbrIdxPointer[tmp] - 1];
      }

      M2[0][0] = 1;
      M2[0][1] = C[NbrIdx[0] * n + NbrIdx[1]];
      M2[0][2] = C[NbrIdx[0] * n + NbrIdx[2]];
      M2[0][3] = C[NbrIdx[0] * n + NbrIdx[3]];
      M2[0][4] = C[NbrIdx[0] * n + NbrIdx[4]];

      M2[1][0] = M2[0][1];
      M2[1][1] = 1;
      M2[1][2] = C[NbrIdx[1] * n + NbrIdx[2]];
      M2[1][3] = C[NbrIdx[1] * n + NbrIdx[3]];
      M2[1][4] = C[NbrIdx[1] * n + NbrIdx[4]];

      M2[2][0] = M2[0][2];
      M2[2][1] = M2[1][2];
      M2[2][2] = 1;
      M2[2][3] = C[NbrIdx[2] * n + NbrIdx[3]];
      M2[2][4] = C[NbrIdx[2] * n + NbrIdx[4]];

      M2[3][0] = M2[0][3];
      M2[3][1] = M2[1][3];
      M2[3][2] = M2[2][3];
      M2[3][3] = 1;
      M2[3][4] = C[NbrIdx[3] * n + NbrIdx[4]];

      M2[4][0] = M2[0][4];
      M2[4][1] = M2[1][4];
      M2[4][2] = M2[2][4];
      M2[4][3] = M2[3][4];
      M2[4][4] = 1;

      M1[0][0] = C[XIdx * n + NbrIdx[0]];
      M1[0][1] = C[XIdx * n + NbrIdx[1]];
      M1[0][2] = C[XIdx * n + NbrIdx[2]];
      M1[0][3] = C[XIdx * n + NbrIdx[3]];
      M1[0][4] = C[XIdx * n + NbrIdx[4]];
      pseudoinversel5(M2, M2Inv, v, rv1, w, res1);
      for (int d2 = 0; d2 < SizeOfArr; d2++) {
        if ((d2 == (NbrIdxPointer[0] - 1)) || (d2 == (NbrIdxPointer[1] - 1)) || (d2 == (NbrIdxPointer[2] - 1)) ||
            (d2 == (NbrIdxPointer[3] - 1)) || (d2 == (NbrIdxPointer[4] - 1))) {
          continue;
        }
        YIdx = G_Chunk[d2];

        if (XIdx < num_heterogeneous_rows || YIdx < num_heterogeneous_rows) {
          continue;
        }

        if (G[XIdx * n + YIdx] == 1) {
          NoEdgeFlag = 0;
          M0 = C[XIdx * n + YIdx];
          // Beginning Of the Indep Test Calculation

          M1[1][0] = C[YIdx * n + NbrIdx[0]];
          M1[1][1] = C[YIdx * n + NbrIdx[1]];
          M1[1][2] = C[YIdx * n + NbrIdx[2]];
          M1[1][3] = C[YIdx * n + NbrIdx[3]];
          M1[1][4] = C[YIdx * n + NbrIdx[4]];
          // Begin to calculate I2Inv Using pseudo-inverse
          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 5; c2++) {
              M1MulM2Inv[c1][c2] = 0;
              for (int c3 = 0; c3 < 5; c3++) M1MulM2Inv[c1][c2] += M1[c1][c3] * M2Inv[c3][c2];
            }
          }

          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 2; c2++) {
              H[c1][c2] = 0;
              for (int c3 = 0; c3 < 5; c3++) H[c1][c2] += M1MulM2Inv[c1][c3] * M1[c2][c3];
            }
          }
          H[0][0] = 1 - H[0][0];
          H[0][1] = M0 - H[0][1];
          H[1][1] = 1 - H[1][1];

          rho = H[0][1] / (sqrt(abs(H[0][0] * H[1][1])));
          Z = abs(0.5 * log(abs((1 + rho) / (1 - rho))));

          if (Z < th) {
            if (atomicCAS(&mutex[XIdx * n + YIdx], 0, 1) == 0) {  // lock
              G[XIdx * n + YIdx] = 0;
              G[YIdx * n + XIdx] = 0;
              pMax[XIdx * n + YIdx] = Z;
              Sepset[(XIdx * n + YIdx) * ML] = NbrIdx[0];
              Sepset[(XIdx * n + YIdx) * ML + 1] = NbrIdx[1];
              Sepset[(XIdx * n + YIdx) * ML + 2] = NbrIdx[2];
              Sepset[(XIdx * n + YIdx) * ML + 3] = NbrIdx[3];
              Sepset[(XIdx * n + YIdx) * ML + 4] = NbrIdx[4];
            }
          }
        }
      }
    }
  }
}

__global__ void cal_Indepl6(double* C, int* G, int* GPrime, int* mutex, int* Sepset, double* pMax, double th, int n,
                            int num_heterogeneous_rows) {
  int YIdx;
  int XIdx = blockIdx.y;
  int NbrIdxPointer[6];
  int SizeOfArr;
  int NumberOfJump;
  int NumOfGivenJump;
  int NumOfComb;
  __shared__ int NoEdgeFlag;
  int NbrIdx[6];

  double M0;
  double M1[2][6];
  double M2[6][6];
  double M2Inv[6][6];
  double M1MulM2Inv[2][6];
  double H[2][2];
  double rho;
  double Z;
  extern __shared__ int G_Chunk[];
  // pseudo-inverse parameter
  double v[6][6];
  double w[6], rv1[6];
  double res1[6][6];

  NoEdgeFlag = 0;
  SizeOfArr = GPrime[XIdx * n + n - 1];
  if (SizeOfArr <= 6) {
    return;
  }

  if ((SizeOfArr % ParGivenL6) == 0) {
    NumberOfJump = SizeOfArr / ParGivenL6;
  } else {
    NumberOfJump = SizeOfArr / ParGivenL6 + 1;
  }
  // Copy Row Xid from GPrime to G_chunck
  for (int cnt = 0; cnt < NumberOfJump; cnt++) {
    if ((threadIdx.x + cnt * ParGivenL6) < SizeOfArr) {
      G_Chunk[threadIdx.x + cnt * ParGivenL6] = GPrime[XIdx * n + threadIdx.x + cnt * ParGivenL6];
    }
    __syncthreads();
  }

  BINOM(SizeOfArr, 6, &NumOfComb);
  if ((NumOfComb % (ParGivenL6 * NumOfBlockForEachNodeL6)) == 0) {
    NumOfGivenJump = NumOfComb / (ParGivenL6 * NumOfBlockForEachNodeL6);
  } else {
    NumOfGivenJump = NumOfComb / (ParGivenL6 * NumOfBlockForEachNodeL6) + 1;
  }

  for (int d1 = 0; d1 < NumOfGivenJump; d1++) {
    __syncthreads();
    if (NoEdgeFlag == 1) {
      return;
    }
    if (threadIdx.x + blockIdx.x * ParGivenL6 + d1 * ParGivenL6 * NumOfBlockForEachNodeL6 < NumOfComb) {
      __syncthreads();
      NoEdgeFlag = 1;
      __syncthreads();
      IthCombination(NbrIdxPointer, SizeOfArr, 6,
                     threadIdx.x + blockIdx.x * ParGivenL6 + d1 * ParGivenL6 * NumOfBlockForEachNodeL6 + 1);
      for (int tmp = 0; tmp < 6; tmp++) {
        NbrIdx[tmp] = G_Chunk[NbrIdxPointer[tmp] - 1];
      }

      M2[0][0] = 1;
      M2[0][1] = C[NbrIdx[0] * n + NbrIdx[1]];
      M2[0][2] = C[NbrIdx[0] * n + NbrIdx[2]];
      M2[0][3] = C[NbrIdx[0] * n + NbrIdx[3]];
      M2[0][4] = C[NbrIdx[0] * n + NbrIdx[4]];
      M2[0][5] = C[NbrIdx[0] * n + NbrIdx[5]];

      M2[1][0] = M2[0][1];
      M2[1][1] = 1;
      M2[1][2] = C[NbrIdx[1] * n + NbrIdx[2]];
      M2[1][3] = C[NbrIdx[1] * n + NbrIdx[3]];
      M2[1][4] = C[NbrIdx[1] * n + NbrIdx[4]];
      M2[1][5] = C[NbrIdx[1] * n + NbrIdx[5]];

      M2[2][0] = M2[0][2];
      M2[2][1] = M2[1][2];
      M2[2][2] = 1;
      M2[2][3] = C[NbrIdx[2] * n + NbrIdx[3]];
      M2[2][4] = C[NbrIdx[2] * n + NbrIdx[4]];
      M2[2][5] = C[NbrIdx[2] * n + NbrIdx[5]];

      M2[3][0] = M2[0][3];
      M2[3][1] = M2[1][3];
      M2[3][2] = M2[2][3];
      M2[3][3] = 1;
      M2[3][4] = C[NbrIdx[3] * n + NbrIdx[4]];
      M2[3][5] = C[NbrIdx[3] * n + NbrIdx[5]];

      M2[4][0] = M2[0][4];
      M2[4][1] = M2[1][4];
      M2[4][2] = M2[2][4];
      M2[4][3] = M2[3][4];
      M2[4][4] = 1;
      M2[4][5] = C[NbrIdx[4] * n + NbrIdx[5]];

      M2[5][0] = M2[0][5];
      M2[5][1] = M2[1][5];
      M2[5][2] = M2[2][5];
      M2[5][3] = M2[3][5];
      M2[5][4] = M2[4][5];
      M2[5][5] = 1;

      M1[0][0] = C[XIdx * n + NbrIdx[0]];
      M1[0][1] = C[XIdx * n + NbrIdx[1]];
      M1[0][2] = C[XIdx * n + NbrIdx[2]];
      M1[0][3] = C[XIdx * n + NbrIdx[3]];
      M1[0][4] = C[XIdx * n + NbrIdx[4]];
      M1[0][5] = C[XIdx * n + NbrIdx[5]];

      pseudoinversel6(M2, M2Inv, v, rv1, w, res1);
      for (int d2 = 0; d2 < SizeOfArr; d2++) {
        if ((d2 == (NbrIdxPointer[0] - 1)) || (d2 == (NbrIdxPointer[1] - 1)) || (d2 == (NbrIdxPointer[2] - 1)) ||
            (d2 == (NbrIdxPointer[3] - 1)) || (d2 == (NbrIdxPointer[4] - 1)) || (d2 == (NbrIdxPointer[5] - 1))) {
          continue;
        }
        YIdx = G_Chunk[d2];

        if (XIdx < num_heterogeneous_rows || YIdx < num_heterogeneous_rows) {
          continue;
        }

        if (G[XIdx * n + YIdx] == 1) {
          NoEdgeFlag = 0;
          M0 = C[XIdx * n + YIdx];
          // Beginning Of the Indep Test Calculation
          M1[1][0] = C[YIdx * n + NbrIdx[0]];
          M1[1][1] = C[YIdx * n + NbrIdx[1]];
          M1[1][2] = C[YIdx * n + NbrIdx[2]];
          M1[1][3] = C[YIdx * n + NbrIdx[3]];
          M1[1][4] = C[YIdx * n + NbrIdx[4]];
          M1[1][5] = C[YIdx * n + NbrIdx[5]];
          // Begin to calculate I2Inv Using pseudo-inverse
          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 6; c2++) {
              M1MulM2Inv[c1][c2] = 0;
              for (int c3 = 0; c3 < 6; c3++) M1MulM2Inv[c1][c2] += M1[c1][c3] * M2Inv[c3][c2];
            }
          }

          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 2; c2++) {
              H[c1][c2] = 0;
              for (int c3 = 0; c3 < 6; c3++) H[c1][c2] += M1MulM2Inv[c1][c3] * M1[c2][c3];
            }
          }
          H[0][0] = 1 - H[0][0];
          H[0][1] = M0 - H[0][1];
          H[1][1] = 1 - H[1][1];

          rho = H[0][1] / (sqrt(abs(H[0][0] * H[1][1])));
          Z = abs(0.5 * log(abs((1 + rho) / (1 - rho))));

          if (Z < th) {
            if (atomicCAS(&mutex[XIdx * n + YIdx], 0, 1) == 0) {  // lock
              G[XIdx * n + YIdx] = 0;
              G[YIdx * n + XIdx] = 0;
              pMax[XIdx * n + YIdx] = Z;
              Sepset[(XIdx * n + YIdx) * ML] = NbrIdx[0];
              Sepset[(XIdx * n + YIdx) * ML + 1] = NbrIdx[1];
              Sepset[(XIdx * n + YIdx) * ML + 2] = NbrIdx[2];
              Sepset[(XIdx * n + YIdx) * ML + 3] = NbrIdx[3];
              Sepset[(XIdx * n + YIdx) * ML + 4] = NbrIdx[4];
              Sepset[(XIdx * n + YIdx) * ML + 5] = NbrIdx[5];
            }
          }
        }
      }
    }
  }
}

__global__ void cal_Indepl7(double* C, int* G, int* GPrime, int* mutex, int* Sepset, double* pMax, double th, int n,
                            int num_heterogeneous_rows) {
  int YIdx;
  int XIdx = blockIdx.y;
  int NbrIdxPointer[7];
  int SizeOfArr;
  int NumberOfJump;
  int NumOfGivenJump;
  int NumOfComb;
  __shared__ int NoEdgeFlag;
  int NbrIdx[7];

  double M0;
  double M1[2][7];
  double M2[7][7];
  double M2Inv[7][7];
  double M1MulM2Inv[2][7];
  double H[2][2];
  double rho;
  double Z;
  // pseudo-inverse parameter
  double v[7][7];
  double w[7], rv1[7];
  double res1[7][7];

  extern __shared__ int G_Chunk[];

  NoEdgeFlag = 0;
  SizeOfArr = GPrime[XIdx * n + n - 1];
  if (SizeOfArr <= 7) {
    return;
  }

  if ((SizeOfArr % ParGivenL7) == 0) {
    NumberOfJump = SizeOfArr / ParGivenL7;
  } else {
    NumberOfJump = SizeOfArr / ParGivenL7 + 1;
  }
  // Copy Row Xid from GPrime to G_chunck
  for (int cnt = 0; cnt < NumberOfJump; cnt++) {
    if ((threadIdx.x + cnt * ParGivenL7) < SizeOfArr) {
      G_Chunk[threadIdx.x + cnt * ParGivenL7] = GPrime[XIdx * n + threadIdx.x + cnt * ParGivenL7];
    }
    __syncthreads();
  }

  BINOM(SizeOfArr, 7, &NumOfComb);
  if ((NumOfComb % (ParGivenL7 * NumOfBlockForEachNodeL7)) == 0) {
    NumOfGivenJump = NumOfComb / (ParGivenL7 * NumOfBlockForEachNodeL7);
  } else {
    NumOfGivenJump = NumOfComb / (ParGivenL7 * NumOfBlockForEachNodeL7) + 1;
  }

  for (int d1 = 0; d1 < NumOfGivenJump; d1++) {
    __syncthreads();
    if (NoEdgeFlag == 1) {
      return;
    }
    if (threadIdx.x + blockIdx.x * ParGivenL7 + d1 * ParGivenL7 * NumOfBlockForEachNodeL7 < NumOfComb) {
      __syncthreads();
      NoEdgeFlag = 1;
      __syncthreads();
      IthCombination(NbrIdxPointer, SizeOfArr, 7,
                     threadIdx.x + blockIdx.x * ParGivenL7 + d1 * ParGivenL7 * NumOfBlockForEachNodeL7 + 1);
      for (int tmp = 0; tmp < 7; tmp++) {
        NbrIdx[tmp] = G_Chunk[NbrIdxPointer[tmp] - 1];
      }

      M2[0][0] = 1;
      M2[0][1] = C[NbrIdx[0] * n + NbrIdx[1]];
      M2[0][2] = C[NbrIdx[0] * n + NbrIdx[2]];
      M2[0][3] = C[NbrIdx[0] * n + NbrIdx[3]];
      M2[0][4] = C[NbrIdx[0] * n + NbrIdx[4]];
      M2[0][5] = C[NbrIdx[0] * n + NbrIdx[5]];
      M2[0][6] = C[NbrIdx[0] * n + NbrIdx[6]];

      M2[1][0] = M2[0][1];
      M2[1][1] = 1;
      M2[1][2] = C[NbrIdx[1] * n + NbrIdx[2]];
      M2[1][3] = C[NbrIdx[1] * n + NbrIdx[3]];
      M2[1][4] = C[NbrIdx[1] * n + NbrIdx[4]];
      M2[1][5] = C[NbrIdx[1] * n + NbrIdx[5]];
      M2[1][6] = C[NbrIdx[1] * n + NbrIdx[6]];

      M2[2][0] = M2[0][2];
      M2[2][1] = M2[1][2];
      M2[2][2] = 1;
      M2[2][3] = C[NbrIdx[2] * n + NbrIdx[3]];
      M2[2][4] = C[NbrIdx[2] * n + NbrIdx[4]];
      M2[2][5] = C[NbrIdx[2] * n + NbrIdx[5]];
      M2[2][6] = C[NbrIdx[2] * n + NbrIdx[6]];

      M2[3][0] = M2[0][3];
      M2[3][1] = M2[1][3];
      M2[3][2] = M2[2][3];
      M2[3][3] = 1;
      M2[3][4] = C[NbrIdx[3] * n + NbrIdx[4]];
      M2[3][5] = C[NbrIdx[3] * n + NbrIdx[5]];
      M2[3][6] = C[NbrIdx[3] * n + NbrIdx[6]];

      M2[4][0] = M2[0][4];
      M2[4][1] = M2[1][4];
      M2[4][2] = M2[2][4];
      M2[4][3] = M2[3][4];
      M2[4][4] = 1;
      M2[4][5] = C[NbrIdx[4] * n + NbrIdx[5]];
      M2[4][6] = C[NbrIdx[4] * n + NbrIdx[6]];

      M2[5][0] = M2[0][5];
      M2[5][1] = M2[1][5];
      M2[5][2] = M2[2][5];
      M2[5][3] = M2[3][5];
      M2[5][4] = M2[4][5];
      M2[5][5] = 1;
      M2[5][6] = C[NbrIdx[5] * n + NbrIdx[6]];

      M2[6][0] = M2[0][6];
      M2[6][1] = M2[1][6];
      M2[6][2] = M2[2][6];
      M2[6][3] = M2[3][6];
      M2[6][4] = M2[4][6];
      M2[6][5] = M2[5][6];
      M2[6][6] = 1;
      pseudoinversel7(M2, M2Inv, v, rv1, w, res1);

      M1[0][0] = C[XIdx * n + NbrIdx[0]];
      M1[0][1] = C[XIdx * n + NbrIdx[1]];
      M1[0][2] = C[XIdx * n + NbrIdx[2]];
      M1[0][3] = C[XIdx * n + NbrIdx[3]];
      M1[0][4] = C[XIdx * n + NbrIdx[4]];
      M1[0][5] = C[XIdx * n + NbrIdx[5]];
      M1[0][6] = C[XIdx * n + NbrIdx[6]];

      for (int d2 = 0; d2 < SizeOfArr; d2++) {
        if ((d2 == (NbrIdxPointer[0] - 1)) || (d2 == (NbrIdxPointer[1] - 1)) || (d2 == (NbrIdxPointer[2] - 1)) ||
            (d2 == (NbrIdxPointer[3] - 1)) || (d2 == (NbrIdxPointer[4] - 1)) || (d2 == (NbrIdxPointer[5] - 1)) ||
            (d2 == (NbrIdxPointer[6] - 1))) {
          continue;
        }
        YIdx = G_Chunk[d2];

        if (XIdx < num_heterogeneous_rows || YIdx < num_heterogeneous_rows) {
          continue;
        }

        if (G[XIdx * n + YIdx] == 1) {
          NoEdgeFlag = 0;
          M0 = C[XIdx * n + YIdx];
          // Beginning Of the Indep Test Calculation

          M1[1][0] = C[YIdx * n + NbrIdx[0]];
          M1[1][1] = C[YIdx * n + NbrIdx[1]];
          M1[1][2] = C[YIdx * n + NbrIdx[2]];
          M1[1][3] = C[YIdx * n + NbrIdx[3]];
          M1[1][4] = C[YIdx * n + NbrIdx[4]];
          M1[1][5] = C[YIdx * n + NbrIdx[5]];
          M1[1][6] = C[YIdx * n + NbrIdx[6]];
          // Begin to calculate I2Inv Using pseudo-inverse
          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 7; c2++) {
              M1MulM2Inv[c1][c2] = 0;
              for (int c3 = 0; c3 < 7; c3++) M1MulM2Inv[c1][c2] += M1[c1][c3] * M2Inv[c3][c2];
            }
          }

          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 2; c2++) {
              H[c1][c2] = 0;
              for (int c3 = 0; c3 < 7; c3++) H[c1][c2] += M1MulM2Inv[c1][c3] * M1[c2][c3];
            }
          }
          H[0][0] = 1 - H[0][0];
          H[0][1] = M0 - H[0][1];
          H[1][1] = 1 - H[1][1];

          rho = H[0][1] / (sqrt(abs(H[0][0] * H[1][1])));
          Z = abs(0.5 * log(abs((1 + rho) / (1 - rho))));

          if (Z < th) {
            if (atomicCAS(&mutex[XIdx * n + YIdx], 0, 1) == 0) {  // lock
              G[XIdx * n + YIdx] = 0;
              G[YIdx * n + XIdx] = 0;
              pMax[XIdx * n + YIdx] = Z;
              Sepset[(XIdx * n + YIdx) * ML] = NbrIdx[0];
              Sepset[(XIdx * n + YIdx) * ML + 1] = NbrIdx[1];
              Sepset[(XIdx * n + YIdx) * ML + 2] = NbrIdx[2];
              Sepset[(XIdx * n + YIdx) * ML + 3] = NbrIdx[3];
              Sepset[(XIdx * n + YIdx) * ML + 4] = NbrIdx[4];
              Sepset[(XIdx * n + YIdx) * ML + 5] = NbrIdx[5];
              Sepset[(XIdx * n + YIdx) * ML + 6] = NbrIdx[6];
            }
          }
        }
      }
    }
  }
}

__global__ void cal_Indepl8(double* C, int* G, int* GPrime, int* mutex, int* Sepset, double* pMax, double th, int n,
                            int num_heterogeneous_rows) {
  int YIdx;
  int XIdx = blockIdx.y;
  int NbrIdxPointer[8];
  int SizeOfArr;
  int NumberOfJump;
  int NumOfGivenJump;
  int NumOfComb;
  __shared__ int NoEdgeFlag;
  int NbrIdx[8];

  double M0;
  double M1[2][8];
  double M2[8][8];
  double M2Inv[8][8];
  double M1MulM2Inv[2][8];
  double H[2][2];
  double rho;
  double Z;
  // pseudo-inverse parameter
  double v[8][8];
  double w[8], rv1[8];
  double res1[8][8];

  extern __shared__ int G_Chunk[];

  NoEdgeFlag = 0;
  SizeOfArr = GPrime[XIdx * n + n - 1];
  if (SizeOfArr <= 8) {
    return;
  }

  if ((SizeOfArr % ParGivenL8) == 0) {
    NumberOfJump = SizeOfArr / ParGivenL8;
  } else {
    NumberOfJump = SizeOfArr / ParGivenL8 + 1;
  }
  // Copy Row Xid from GPrime to G_chunck
  for (int cnt = 0; cnt < NumberOfJump; cnt++) {
    if ((threadIdx.x + cnt * ParGivenL8) < SizeOfArr) {
      G_Chunk[threadIdx.x + cnt * ParGivenL8] = GPrime[XIdx * n + threadIdx.x + cnt * ParGivenL8];
    }
    __syncthreads();
  }

  BINOM(SizeOfArr, 8, &NumOfComb);
  if ((NumOfComb % (ParGivenL8 * NumOfBlockForEachNodeL8)) == 0) {
    NumOfGivenJump = NumOfComb / (ParGivenL8 * NumOfBlockForEachNodeL8);
  } else {
    NumOfGivenJump = NumOfComb / (ParGivenL8 * NumOfBlockForEachNodeL8) + 1;
  }

  for (int d1 = 0; d1 < NumOfGivenJump; d1++) {
    __syncthreads();
    if (NoEdgeFlag == 1) {
      return;
    }
    if (threadIdx.x + blockIdx.x * ParGivenL8 + d1 * ParGivenL8 * NumOfBlockForEachNodeL8 < NumOfComb) {
      __syncthreads();
      NoEdgeFlag = 1;
      __syncthreads();
      IthCombination(NbrIdxPointer, SizeOfArr, 8,
                     threadIdx.x + blockIdx.x * ParGivenL8 + d1 * ParGivenL8 * NumOfBlockForEachNodeL8 + 1);
      for (int tmp = 0; tmp < 8; tmp++) {
        NbrIdx[tmp] = G_Chunk[NbrIdxPointer[tmp] - 1];
      }

      M2[0][0] = 1;
      M2[0][1] = C[NbrIdx[0] * n + NbrIdx[1]];
      M2[0][2] = C[NbrIdx[0] * n + NbrIdx[2]];
      M2[0][3] = C[NbrIdx[0] * n + NbrIdx[3]];
      M2[0][4] = C[NbrIdx[0] * n + NbrIdx[4]];
      M2[0][5] = C[NbrIdx[0] * n + NbrIdx[5]];
      M2[0][6] = C[NbrIdx[0] * n + NbrIdx[6]];
      M2[0][7] = C[NbrIdx[0] * n + NbrIdx[7]];

      M2[1][0] = M2[0][1];
      M2[1][1] = 1;
      M2[1][2] = C[NbrIdx[1] * n + NbrIdx[2]];
      M2[1][3] = C[NbrIdx[1] * n + NbrIdx[3]];
      M2[1][4] = C[NbrIdx[1] * n + NbrIdx[4]];
      M2[1][5] = C[NbrIdx[1] * n + NbrIdx[5]];
      M2[1][6] = C[NbrIdx[1] * n + NbrIdx[6]];
      M2[1][7] = C[NbrIdx[1] * n + NbrIdx[7]];

      M2[2][0] = M2[0][2];
      M2[2][1] = M2[1][2];
      M2[2][2] = 1;
      M2[2][3] = C[NbrIdx[2] * n + NbrIdx[3]];
      M2[2][4] = C[NbrIdx[2] * n + NbrIdx[4]];
      M2[2][5] = C[NbrIdx[2] * n + NbrIdx[5]];
      M2[2][6] = C[NbrIdx[2] * n + NbrIdx[6]];
      M2[2][7] = C[NbrIdx[2] * n + NbrIdx[7]];

      M2[3][0] = M2[0][3];
      M2[3][1] = M2[1][3];
      M2[3][2] = M2[2][3];
      M2[3][3] = 1;
      M2[3][4] = C[NbrIdx[3] * n + NbrIdx[4]];
      M2[3][5] = C[NbrIdx[3] * n + NbrIdx[5]];
      M2[3][6] = C[NbrIdx[3] * n + NbrIdx[6]];
      M2[3][7] = C[NbrIdx[3] * n + NbrIdx[7]];

      M2[4][0] = M2[0][4];
      M2[4][1] = M2[1][4];
      M2[4][2] = M2[2][4];
      M2[4][3] = M2[3][4];
      M2[4][4] = 1;
      M2[4][5] = C[NbrIdx[4] * n + NbrIdx[5]];
      M2[4][6] = C[NbrIdx[4] * n + NbrIdx[6]];
      M2[4][7] = C[NbrIdx[4] * n + NbrIdx[7]];

      M2[5][0] = M2[0][5];
      M2[5][1] = M2[1][5];
      M2[5][2] = M2[2][5];
      M2[5][3] = M2[3][5];
      M2[5][4] = M2[4][5];
      M2[5][5] = 1;
      M2[5][6] = C[NbrIdx[5] * n + NbrIdx[6]];
      M2[5][7] = C[NbrIdx[5] * n + NbrIdx[7]];

      M2[6][0] = M2[0][6];
      M2[6][1] = M2[1][6];
      M2[6][2] = M2[2][6];
      M2[6][3] = M2[3][6];
      M2[6][4] = M2[4][6];
      M2[6][5] = M2[5][6];
      M2[6][6] = 1;
      M2[6][7] = C[NbrIdx[6] * n + NbrIdx[7]];

      M2[7][0] = M2[0][7];
      M2[7][1] = M2[1][7];
      M2[7][2] = M2[2][7];
      M2[7][3] = M2[3][7];
      M2[7][4] = M2[4][7];
      M2[7][5] = M2[5][7];
      M2[7][6] = M2[6][7];
      M2[7][7] = 1;

      M1[0][0] = C[XIdx * n + NbrIdx[0]];
      M1[0][1] = C[XIdx * n + NbrIdx[1]];
      M1[0][2] = C[XIdx * n + NbrIdx[2]];
      M1[0][3] = C[XIdx * n + NbrIdx[3]];
      M1[0][4] = C[XIdx * n + NbrIdx[4]];
      M1[0][5] = C[XIdx * n + NbrIdx[5]];
      M1[0][6] = C[XIdx * n + NbrIdx[6]];
      M1[0][7] = C[XIdx * n + NbrIdx[7]];
      pseudoinversel8(M2, M2Inv, v, rv1, w, res1);
      for (int d2 = 0; d2 < SizeOfArr; d2++) {
        if ((d2 == (NbrIdxPointer[0] - 1)) || (d2 == (NbrIdxPointer[1] - 1)) || (d2 == (NbrIdxPointer[2] - 1)) ||
            (d2 == (NbrIdxPointer[3] - 1)) || (d2 == (NbrIdxPointer[4] - 1)) || (d2 == (NbrIdxPointer[5] - 1)) ||
            (d2 == (NbrIdxPointer[6] - 1)) || (d2 == (NbrIdxPointer[7] - 1))) {
          continue;
        }
        YIdx = G_Chunk[d2];

        if (XIdx < num_heterogeneous_rows || YIdx < num_heterogeneous_rows) {
          continue;
        }

        if (G[XIdx * n + YIdx] == 1) {
          NoEdgeFlag = 0;
          M0 = C[XIdx * n + YIdx];
          // Beginning Of the Indep Test Calculation
          M1[1][0] = C[YIdx * n + NbrIdx[0]];
          M1[1][1] = C[YIdx * n + NbrIdx[1]];
          M1[1][2] = C[YIdx * n + NbrIdx[2]];
          M1[1][3] = C[YIdx * n + NbrIdx[3]];
          M1[1][4] = C[YIdx * n + NbrIdx[4]];
          M1[1][5] = C[YIdx * n + NbrIdx[5]];
          M1[1][6] = C[YIdx * n + NbrIdx[6]];
          M1[1][7] = C[YIdx * n + NbrIdx[7]];
          // Begin to calculate I2Inv Using pseudo-inverse
          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 8; c2++) {
              M1MulM2Inv[c1][c2] = 0;
              for (int c3 = 0; c3 < 8; c3++) M1MulM2Inv[c1][c2] += M1[c1][c3] * M2Inv[c3][c2];
            }
          }

          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 2; c2++) {
              H[c1][c2] = 0;
              for (int c3 = 0; c3 < 8; c3++) H[c1][c2] += M1MulM2Inv[c1][c3] * M1[c2][c3];
            }
          }
          H[0][0] = 1 - H[0][0];
          H[0][1] = M0 - H[0][1];
          H[1][1] = 1 - H[1][1];

          rho = H[0][1] / (sqrt(abs(H[0][0] * H[1][1])));
          Z = abs(0.5 * log(abs((1 + rho) / (1 - rho))));

          if (Z < th) {
            if (atomicCAS(&mutex[XIdx * n + YIdx], 0, 1) == 0) {  // lock
              G[XIdx * n + YIdx] = 0;
              G[YIdx * n + XIdx] = 0;
              pMax[XIdx * n + YIdx] = Z;
              Sepset[(XIdx * n + YIdx) * ML] = NbrIdx[0];
              Sepset[(XIdx * n + YIdx) * ML + 1] = NbrIdx[1];
              Sepset[(XIdx * n + YIdx) * ML + 2] = NbrIdx[2];
              Sepset[(XIdx * n + YIdx) * ML + 3] = NbrIdx[3];
              Sepset[(XIdx * n + YIdx) * ML + 4] = NbrIdx[4];
              Sepset[(XIdx * n + YIdx) * ML + 5] = NbrIdx[5];
              Sepset[(XIdx * n + YIdx) * ML + 6] = NbrIdx[6];
              Sepset[(XIdx * n + YIdx) * ML + 7] = NbrIdx[7];
            }
          }
        }
      }
    }
  }
}

__global__ void cal_Indepl9(double* C, int* G, int* GPrime, int* mutex, int* Sepset, double* pMax, double th, int n,
                            int num_heterogeneous_rows) {
  int YIdx;
  int XIdx = blockIdx.y;
  int NbrIdxPointer[9];
  int SizeOfArr;
  int NumberOfJump;
  int NumOfGivenJump;
  int NumOfComb;
  __shared__ int NoEdgeFlag;
  int NbrIdx[9];

  double M0;
  double M1[2][9];
  double M2[9][9];
  double M2Inv[9][9];
  double M1MulM2Inv[2][9];
  double H[2][2];
  double rho;
  double Z;
  // pseudo-inverse parameter
  double v[9][9];
  double w[9], rv1[9];
  double res1[9][9];

  extern __shared__ int G_Chunk[];

  NoEdgeFlag = 0;
  SizeOfArr = GPrime[XIdx * n + n - 1];
  if (SizeOfArr <= 9) {
    return;
  }
  if ((SizeOfArr % ParGivenL9) == 0) {
    NumberOfJump = SizeOfArr / ParGivenL9;
  } else {
    NumberOfJump = SizeOfArr / ParGivenL9 + 1;
  }
  // Copy Row Xid from GPrime to G_chunck
  for (int cnt = 0; cnt < NumberOfJump; cnt++) {
    if ((threadIdx.x + cnt * ParGivenL9) < SizeOfArr) {
      G_Chunk[threadIdx.x + cnt * ParGivenL9] = GPrime[XIdx * n + threadIdx.x + cnt * ParGivenL9];
    }
    __syncthreads();
  }

  BINOM(SizeOfArr, 9, &NumOfComb);
  if ((NumOfComb % (ParGivenL9 * NumOfBlockForEachNodeL9)) == 0) {
    NumOfGivenJump = NumOfComb / (ParGivenL9 * NumOfBlockForEachNodeL9);
  } else {
    NumOfGivenJump = NumOfComb / (ParGivenL9 * NumOfBlockForEachNodeL9) + 1;
  }

  for (int d1 = 0; d1 < NumOfGivenJump; d1++) {
    __syncthreads();
    if (NoEdgeFlag == 1) {
      return;
    }
    if (threadIdx.x + blockIdx.x * ParGivenL9 + d1 * ParGivenL9 * NumOfBlockForEachNodeL9 < NumOfComb) {
      __syncthreads();
      NoEdgeFlag = 1;
      __syncthreads();
      IthCombination(NbrIdxPointer, SizeOfArr, 9,
                     threadIdx.x + blockIdx.x * ParGivenL9 + d1 * ParGivenL9 * NumOfBlockForEachNodeL9 + 1);
      for (int tmp = 0; tmp < 9; tmp++) {
        NbrIdx[tmp] = G_Chunk[NbrIdxPointer[tmp] - 1];
      }

      for (int c1 = 0; c1 < 9; c1++) {
        for (int c2 = 0; c2 < 9; c2++) {
          if (c1 > c2) {
            M2[c1][c2] = M2[c2][c1];
          } else if (c1 == c2) {
            M2[c1][c1] = 1;
          } else {
            M2[c1][c2] = C[NbrIdx[c1] * n + NbrIdx[c2]];
          }
        }
      }

      for (int c1 = 0; c1 < 9; c1++) {
        M1[0][c1] = C[XIdx * n + NbrIdx[c1]];
      }

      pseudoinversel9(M2, M2Inv, v, rv1, w, res1);
      for (int d2 = 0; d2 < SizeOfArr; d2++) {
        if ((d2 == (NbrIdxPointer[0] - 1)) || (d2 == (NbrIdxPointer[1] - 1)) || (d2 == (NbrIdxPointer[2] - 1)) ||
            (d2 == (NbrIdxPointer[3] - 1)) || (d2 == (NbrIdxPointer[4] - 1)) || (d2 == (NbrIdxPointer[5] - 1)) ||
            (d2 == (NbrIdxPointer[6] - 1)) || (d2 == (NbrIdxPointer[7] - 1)) || (d2 == (NbrIdxPointer[8] - 1))) {
          continue;
        }
        YIdx = G_Chunk[d2];

        if (XIdx < num_heterogeneous_rows || YIdx < num_heterogeneous_rows) {
          continue;
        }

        if (G[XIdx * n + YIdx] == 1) {
          NoEdgeFlag = 0;
          M0 = C[XIdx * n + YIdx];
          // Beginning Of the Indep Test Calculation
          for (int c1 = 0; c1 < 9; c1++) {
            M1[1][c1] = C[YIdx * n + NbrIdx[c1]];
          }
          // Begin to calculate I2Inv Using pseudo-inverse
          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 9; c2++) {
              M1MulM2Inv[c1][c2] = 0;
              for (int c3 = 0; c3 < 9; c3++) M1MulM2Inv[c1][c2] += M1[c1][c3] * M2Inv[c3][c2];
            }
          }

          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 2; c2++) {
              H[c1][c2] = 0;
              for (int c3 = 0; c3 < 9; c3++) H[c1][c2] += M1MulM2Inv[c1][c3] * M1[c2][c3];
            }
          }
          H[0][0] = 1 - H[0][0];
          H[0][1] = M0 - H[0][1];
          H[1][1] = 1 - H[1][1];

          rho = H[0][1] / (sqrt(abs(H[0][0] * H[1][1])));
          Z = abs(0.5 * log(abs((1 + rho) / (1 - rho))));

          if (Z < th) {
            if (atomicCAS(&mutex[XIdx * n + YIdx], 0, 1) == 0) {  // lock
              G[XIdx * n + YIdx] = 0;
              G[YIdx * n + XIdx] = 0;
              pMax[XIdx * n + YIdx] = Z;
              Sepset[(XIdx * n + YIdx) * ML] = NbrIdx[0];
              Sepset[(XIdx * n + YIdx) * ML + 1] = NbrIdx[1];
              Sepset[(XIdx * n + YIdx) * ML + 2] = NbrIdx[2];
              Sepset[(XIdx * n + YIdx) * ML + 3] = NbrIdx[3];
              Sepset[(XIdx * n + YIdx) * ML + 4] = NbrIdx[4];
              Sepset[(XIdx * n + YIdx) * ML + 5] = NbrIdx[5];
              Sepset[(XIdx * n + YIdx) * ML + 6] = NbrIdx[6];
              Sepset[(XIdx * n + YIdx) * ML + 7] = NbrIdx[7];
              Sepset[(XIdx * n + YIdx) * ML + 8] = NbrIdx[8];
            }
          }
        }
      }
    }
  }
}

__global__ void cal_Indepl10(double* C, int* G, int* GPrime, int* mutex, int* Sepset, double* pMax, double th, int n,
                             int num_heterogeneous_rows) {
  int YIdx;
  int XIdx = blockIdx.y;
  int NbrIdxPointer[10];
  int SizeOfArr;
  int NumberOfJump;
  int NumOfGivenJump;
  int NumOfComb;
  __shared__ int NoEdgeFlag;
  int NbrIdx[10];

  double M0;
  double M1[2][10];
  double M2[10][10];
  double M2Inv[10][10];
  double M1MulM2Inv[2][10];
  double H[2][2];
  double rho;
  double Z;
  // pseudo-inverse parameter
  double v[10][10];
  double w[10], rv1[10];
  double res1[10][10];

  extern __shared__ int G_Chunk[];

  NoEdgeFlag = 0;
  SizeOfArr = GPrime[XIdx * n + n - 1];
  if (SizeOfArr <= 10) {
    return;
  }

  if ((SizeOfArr % ParGivenL10) == 0) {
    NumberOfJump = SizeOfArr / ParGivenL10;
  } else {
    NumberOfJump = SizeOfArr / ParGivenL10 + 1;
  }
  // Copy Row Xid from GPrime to G_chunck
  for (int cnt = 0; cnt < NumberOfJump; cnt++) {
    if ((threadIdx.x + cnt * ParGivenL10) < SizeOfArr) {
      G_Chunk[threadIdx.x + cnt * ParGivenL10] = GPrime[XIdx * n + threadIdx.x + cnt * ParGivenL10];
    }
    __syncthreads();
  }

  BINOM(SizeOfArr, 10, &NumOfComb);
  if ((NumOfComb % (ParGivenL10 * NumOfBlockForEachNodeL10)) == 0) {
    NumOfGivenJump = NumOfComb / (ParGivenL10 * NumOfBlockForEachNodeL10);
  } else {
    NumOfGivenJump = NumOfComb / (ParGivenL10 * NumOfBlockForEachNodeL10) + 1;
  }

  for (int d1 = 0; d1 < NumOfGivenJump; d1++) {
    __syncthreads();
    if (NoEdgeFlag == 1) {
      return;
    }
    if (threadIdx.x + blockIdx.x * ParGivenL10 + d1 * ParGivenL10 * NumOfBlockForEachNodeL10 < NumOfComb) {
      __syncthreads();
      NoEdgeFlag = 1;
      __syncthreads();
      IthCombination(NbrIdxPointer, SizeOfArr, 10,
                     threadIdx.x + blockIdx.x * ParGivenL10 + d1 * ParGivenL10 * NumOfBlockForEachNodeL10 + 1);
      for (int tmp = 0; tmp < 10; tmp++) {
        NbrIdx[tmp] = G_Chunk[NbrIdxPointer[tmp] - 1];
      }

      for (int c1 = 0; c1 < 10; c1++) {
        for (int c2 = 0; c2 < 10; c2++) {
          if (c1 > c2) {
            M2[c1][c2] = M2[c2][c1];
          } else if (c1 == c2) {
            M2[c1][c1] = 1;
          } else {
            M2[c1][c2] = C[NbrIdx[c1] * n + NbrIdx[c2]];
          }
        }
      }

      for (int c1 = 0; c1 < 10; c1++) {
        M1[0][c1] = C[XIdx * n + NbrIdx[c1]];
      }

      pseudoinversel10(M2, M2Inv, v, rv1, w, res1);
      for (int d2 = 0; d2 < SizeOfArr; d2++) {
        if ((d2 == (NbrIdxPointer[0] - 1)) || (d2 == (NbrIdxPointer[1] - 1)) || (d2 == (NbrIdxPointer[2] - 1)) ||
            (d2 == (NbrIdxPointer[3] - 1)) || (d2 == (NbrIdxPointer[4] - 1)) || (d2 == (NbrIdxPointer[5] - 1)) ||
            (d2 == (NbrIdxPointer[6] - 1)) || (d2 == (NbrIdxPointer[7] - 1)) || (d2 == (NbrIdxPointer[8] - 1)) ||
            (d2 == (NbrIdxPointer[9] - 1))) {
          continue;
        }
        YIdx = G_Chunk[d2];

        if (XIdx < num_heterogeneous_rows || YIdx < num_heterogeneous_rows) {
          continue;
        }

        if (G[XIdx * n + YIdx] == 1) {
          NoEdgeFlag = 0;
          M0 = C[XIdx * n + YIdx];
          // Beginning Of the Indep Test Calculation
          for (int c1 = 0; c1 < 10; c1++) {
            M1[1][c1] = C[YIdx * n + NbrIdx[c1]];
          }
          // Begin to calculate I2Inv Using pseudo-inverse
          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 10; c2++) {
              M1MulM2Inv[c1][c2] = 0;
              for (int c3 = 0; c3 < 10; c3++) M1MulM2Inv[c1][c2] += M1[c1][c3] * M2Inv[c3][c2];
            }
          }

          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 2; c2++) {
              H[c1][c2] = 0;
              for (int c3 = 0; c3 < 10; c3++) H[c1][c2] += M1MulM2Inv[c1][c3] * M1[c2][c3];
            }
          }
          H[0][0] = 1 - H[0][0];
          H[0][1] = M0 - H[0][1];
          H[1][1] = 1 - H[1][1];

          rho = H[0][1] / (sqrt(abs(H[0][0] * H[1][1])));
          Z = abs(0.5 * log(abs((1 + rho) / (1 - rho))));

          if (Z < th) {
            if (atomicCAS(&mutex[XIdx * n + YIdx], 0, 1) == 0) {  // lock
              G[XIdx * n + YIdx] = 0;
              G[YIdx * n + XIdx] = 0;
              pMax[XIdx * n + YIdx] = Z;
              Sepset[(XIdx * n + YIdx) * ML] = NbrIdx[0];
              Sepset[(XIdx * n + YIdx) * ML + 1] = NbrIdx[1];
              Sepset[(XIdx * n + YIdx) * ML + 2] = NbrIdx[2];
              Sepset[(XIdx * n + YIdx) * ML + 3] = NbrIdx[3];
              Sepset[(XIdx * n + YIdx) * ML + 4] = NbrIdx[4];
              Sepset[(XIdx * n + YIdx) * ML + 5] = NbrIdx[5];
              Sepset[(XIdx * n + YIdx) * ML + 6] = NbrIdx[6];
              Sepset[(XIdx * n + YIdx) * ML + 7] = NbrIdx[7];
              Sepset[(XIdx * n + YIdx) * ML + 8] = NbrIdx[8];
              Sepset[(XIdx * n + YIdx) * ML + 9] = NbrIdx[9];
            }
          }
        }
      }
    }
  }
}

__global__ void cal_Indepl11(double* C, int* G, int* GPrime, int* mutex, int* Sepset, double* pMax, double th, int n,
                             int num_heterogeneous_rows) {
  int YIdx;
  int XIdx = blockIdx.y;
  int NbrIdxPointer[11];
  int SizeOfArr;
  int NumberOfJump;
  int NumOfGivenJump;
  int NumOfComb;
  __shared__ int NoEdgeFlag;
  int NbrIdx[11];

  double M0;
  double M1[2][11];
  double M2[11][11];
  double M2Inv[11][11];
  double M1MulM2Inv[2][11];
  double H[2][2];
  double rho;
  double Z;
  // pseudo-inverse parameter
  double v[11][11];
  double w[11], rv1[11];
  double res1[11][11];

  extern __shared__ int G_Chunk[];

  NoEdgeFlag = 0;
  SizeOfArr = GPrime[XIdx * n + n - 1];
  if (SizeOfArr <= 11) {
    return;
  }

  if ((SizeOfArr % ParGivenL11) == 0) {
    NumberOfJump = SizeOfArr / ParGivenL11;
  } else {
    NumberOfJump = SizeOfArr / ParGivenL11 + 1;
  }
  // Copy Row Xid from GPrime to G_chunck
  for (int cnt = 0; cnt < NumberOfJump; cnt++) {
    if ((threadIdx.x + cnt * ParGivenL11) < SizeOfArr) {
      G_Chunk[threadIdx.x + cnt * ParGivenL11] = GPrime[XIdx * n + threadIdx.x + cnt * ParGivenL11];
    }
    __syncthreads();
  }

  BINOM(SizeOfArr, 11, &NumOfComb);
  if ((NumOfComb % (ParGivenL11 * NumOfBlockForEachNodeL11)) == 0) {
    NumOfGivenJump = NumOfComb / (ParGivenL11 * NumOfBlockForEachNodeL11);
  } else {
    NumOfGivenJump = NumOfComb / (ParGivenL11 * NumOfBlockForEachNodeL11) + 1;
  }

  for (int d1 = 0; d1 < NumOfGivenJump; d1++) {
    __syncthreads();
    if (NoEdgeFlag == 1) {
      return;
    }
    if (threadIdx.x + blockIdx.x * ParGivenL11 + d1 * ParGivenL11 * NumOfBlockForEachNodeL11 < NumOfComb) {
      __syncthreads();
      NoEdgeFlag = 1;
      __syncthreads();
      IthCombination(NbrIdxPointer, SizeOfArr, 11,
                     threadIdx.x + blockIdx.x * ParGivenL11 + d1 * ParGivenL11 * NumOfBlockForEachNodeL11 + 1);
      for (int tmp = 0; tmp < 11; tmp++) {
        NbrIdx[tmp] = G_Chunk[NbrIdxPointer[tmp] - 1];
      }

      for (int c1 = 0; c1 < 11; c1++) {
        for (int c2 = 0; c2 < 11; c2++) {
          if (c1 > c2) {
            M2[c1][c2] = M2[c2][c1];
          } else if (c1 == c2) {
            M2[c1][c1] = 1;
          } else {
            M2[c1][c2] = C[NbrIdx[c1] * n + NbrIdx[c2]];
          }
        }
      }

      for (int c1 = 0; c1 < 11; c1++) {
        M1[0][c1] = C[XIdx * n + NbrIdx[c1]];
      }

      pseudoinversel11(M2, M2Inv, v, rv1, w, res1);
      for (int d2 = 0; d2 < SizeOfArr; d2++) {
        if ((d2 == (NbrIdxPointer[0] - 1)) || (d2 == (NbrIdxPointer[1] - 1)) || (d2 == (NbrIdxPointer[2] - 1)) ||
            (d2 == (NbrIdxPointer[3] - 1)) || (d2 == (NbrIdxPointer[4] - 1)) || (d2 == (NbrIdxPointer[5] - 1)) ||
            (d2 == (NbrIdxPointer[6] - 1)) || (d2 == (NbrIdxPointer[7] - 1)) || (d2 == (NbrIdxPointer[8] - 1)) ||
            (d2 == (NbrIdxPointer[9] - 1)) || (d2 == (NbrIdxPointer[10] - 1))) {
          continue;
        }
        YIdx = G_Chunk[d2];

        if (XIdx < num_heterogeneous_rows || YIdx < num_heterogeneous_rows) {
          continue;
        }

        if (G[XIdx * n + YIdx] == 1) {
          NoEdgeFlag = 0;
          M0 = C[XIdx * n + YIdx];
          // Beginning Of the Indep Test Calculation
          for (int c1 = 0; c1 < 11; c1++) {
            M1[1][c1] = C[YIdx * n + NbrIdx[c1]];
          }
          // Begin to calculate I2Inv Using pseudo-inverse
          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 11; c2++) {
              M1MulM2Inv[c1][c2] = 0;
              for (int c3 = 0; c3 < 11; c3++) M1MulM2Inv[c1][c2] += M1[c1][c3] * M2Inv[c3][c2];
            }
          }

          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 2; c2++) {
              H[c1][c2] = 0;
              for (int c3 = 0; c3 < 11; c3++) H[c1][c2] += M1MulM2Inv[c1][c3] * M1[c2][c3];
            }
          }
          H[0][0] = 1 - H[0][0];
          H[0][1] = M0 - H[0][1];
          H[1][1] = 1 - H[1][1];

          rho = H[0][1] / (sqrt(abs(H[0][0] * H[1][1])));
          Z = abs(0.5 * log(abs((1 + rho) / (1 - rho))));

          if (Z < th) {
            if (atomicCAS(&mutex[XIdx * n + YIdx], 0, 1) == 0) {  // lock
              G[XIdx * n + YIdx] = 0;
              G[YIdx * n + XIdx] = 0;
              pMax[XIdx * n + YIdx] = Z;
              Sepset[(XIdx * n + YIdx) * ML] = NbrIdx[0];
              Sepset[(XIdx * n + YIdx) * ML + 1] = NbrIdx[1];
              Sepset[(XIdx * n + YIdx) * ML + 2] = NbrIdx[2];
              Sepset[(XIdx * n + YIdx) * ML + 3] = NbrIdx[3];
              Sepset[(XIdx * n + YIdx) * ML + 4] = NbrIdx[4];
              Sepset[(XIdx * n + YIdx) * ML + 5] = NbrIdx[5];
              Sepset[(XIdx * n + YIdx) * ML + 6] = NbrIdx[6];
              Sepset[(XIdx * n + YIdx) * ML + 7] = NbrIdx[7];
              Sepset[(XIdx * n + YIdx) * ML + 8] = NbrIdx[8];
              Sepset[(XIdx * n + YIdx) * ML + 9] = NbrIdx[9];
              Sepset[(XIdx * n + YIdx) * ML + 10] = NbrIdx[10];
            }
          }
        }
      }
    }
  }
}

__global__ void cal_Indepl12(double* C, int* G, int* GPrime, int* mutex, int* Sepset, double* pMax, double th, int n,
                             int num_heterogeneous_rows) {
  int YIdx;
  int XIdx = blockIdx.y;
  int NbrIdxPointer[12];
  int SizeOfArr;
  int NumberOfJump;
  int NumOfGivenJump;
  int NumOfComb;
  __shared__ int NoEdgeFlag;
  int NbrIdx[12];

  double M0;
  double M1[2][12];
  double M2[12][12];
  double M2Inv[12][12];
  double M1MulM2Inv[2][12];
  double H[2][2];
  double rho;
  double Z;
  // pseudo-inverse parameter
  double v[12][12];
  double w[12], rv1[12];
  double res1[12][12];

  extern __shared__ int G_Chunk[];

  NoEdgeFlag = 0;
  SizeOfArr = GPrime[XIdx * n + n - 1];
  if (SizeOfArr <= 12) {
    return;
  }

  if ((SizeOfArr % ParGivenL12) == 0) {
    NumberOfJump = SizeOfArr / ParGivenL12;
  } else {
    NumberOfJump = SizeOfArr / ParGivenL12 + 1;
  }
  // Copy Row Xid from GPrime to G_chunck
  for (int cnt = 0; cnt < NumberOfJump; cnt++) {
    if ((threadIdx.x + cnt * ParGivenL12) < SizeOfArr) {
      G_Chunk[threadIdx.x + cnt * ParGivenL12] = GPrime[XIdx * n + threadIdx.x + cnt * ParGivenL12];
    }
    __syncthreads();
  }

  BINOM(SizeOfArr, 12, &NumOfComb);
  if ((NumOfComb % (ParGivenL12 * NumOfBlockForEachNodeL12)) == 0) {
    NumOfGivenJump = NumOfComb / (ParGivenL12 * NumOfBlockForEachNodeL12);
  } else {
    NumOfGivenJump = NumOfComb / (ParGivenL12 * NumOfBlockForEachNodeL12) + 1;
  }

  for (int d1 = 0; d1 < NumOfGivenJump; d1++) {
    __syncthreads();
    if (NoEdgeFlag == 1) {
      return;
    }
    if (threadIdx.x + blockIdx.x * ParGivenL12 + d1 * ParGivenL12 * NumOfBlockForEachNodeL12 < NumOfComb) {
      __syncthreads();
      NoEdgeFlag = 1;
      __syncthreads();
      IthCombination(NbrIdxPointer, SizeOfArr, 12,
                     threadIdx.x + blockIdx.x * ParGivenL12 + d1 * ParGivenL12 * NumOfBlockForEachNodeL12 + 1);
      for (int tmp = 0; tmp < 12; tmp++) {
        NbrIdx[tmp] = G_Chunk[NbrIdxPointer[tmp] - 1];
      }

      for (int c1 = 0; c1 < 12; c1++) {
        for (int c2 = 0; c2 < 12; c2++) {
          if (c1 > c2) {
            M2[c1][c2] = M2[c2][c1];
          } else if (c1 == c2) {
            M2[c1][c1] = 1;
          } else {
            M2[c1][c2] = C[NbrIdx[c1] * n + NbrIdx[c2]];
          }
        }
      }

      for (int c1 = 0; c1 < 12; c1++) {
        M1[0][c1] = C[XIdx * n + NbrIdx[c1]];
      }

      pseudoinversel12(M2, M2Inv, v, rv1, w, res1);
      for (int d2 = 0; d2 < SizeOfArr; d2++) {
        if ((d2 == (NbrIdxPointer[0] - 1)) || (d2 == (NbrIdxPointer[1] - 1)) || (d2 == (NbrIdxPointer[2] - 1)) ||
            (d2 == (NbrIdxPointer[3] - 1)) || (d2 == (NbrIdxPointer[4] - 1)) || (d2 == (NbrIdxPointer[5] - 1)) ||
            (d2 == (NbrIdxPointer[6] - 1)) || (d2 == (NbrIdxPointer[7] - 1)) || (d2 == (NbrIdxPointer[8] - 1)) ||
            (d2 == (NbrIdxPointer[9] - 1)) || (d2 == (NbrIdxPointer[10] - 1)) || (d2 == (NbrIdxPointer[11] - 1))) {
          continue;
        }
        YIdx = G_Chunk[d2];

        if (XIdx < num_heterogeneous_rows || YIdx < num_heterogeneous_rows) {
          continue;
        }

        if (G[XIdx * n + YIdx] == 1) {
          NoEdgeFlag = 0;
          M0 = C[XIdx * n + YIdx];
          // Beginning Of the Indep Test Calculation
          for (int c1 = 0; c1 < 12; c1++) {
            M1[1][c1] = C[YIdx * n + NbrIdx[c1]];
          }
          // Begin to calculate I2Inv Using pseudo-inverse
          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 12; c2++) {
              M1MulM2Inv[c1][c2] = 0;
              for (int c3 = 0; c3 < 12; c3++) M1MulM2Inv[c1][c2] += M1[c1][c3] * M2Inv[c3][c2];
            }
          }

          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 2; c2++) {
              H[c1][c2] = 0;
              for (int c3 = 0; c3 < 12; c3++) H[c1][c2] += M1MulM2Inv[c1][c3] * M1[c2][c3];
            }
          }
          H[0][0] = 1 - H[0][0];
          H[0][1] = M0 - H[0][1];
          H[1][1] = 1 - H[1][1];

          rho = H[0][1] / (sqrt(abs(H[0][0] * H[1][1])));
          Z = abs(0.5 * log(abs((1 + rho) / (1 - rho))));

          if (Z < th) {
            if (atomicCAS(&mutex[XIdx * n + YIdx], 0, 1) == 0) {  // lock
              G[XIdx * n + YIdx] = 0;
              G[YIdx * n + XIdx] = 0;
              pMax[XIdx * n + YIdx] = Z;
              Sepset[(XIdx * n + YIdx) * ML] = NbrIdx[0];
              Sepset[(XIdx * n + YIdx) * ML + 1] = NbrIdx[1];
              Sepset[(XIdx * n + YIdx) * ML + 2] = NbrIdx[2];
              Sepset[(XIdx * n + YIdx) * ML + 3] = NbrIdx[3];
              Sepset[(XIdx * n + YIdx) * ML + 4] = NbrIdx[4];
              Sepset[(XIdx * n + YIdx) * ML + 5] = NbrIdx[5];
              Sepset[(XIdx * n + YIdx) * ML + 6] = NbrIdx[6];
              Sepset[(XIdx * n + YIdx) * ML + 7] = NbrIdx[7];
              Sepset[(XIdx * n + YIdx) * ML + 8] = NbrIdx[8];
              Sepset[(XIdx * n + YIdx) * ML + 9] = NbrIdx[9];
              Sepset[(XIdx * n + YIdx) * ML + 10] = NbrIdx[10];
              Sepset[(XIdx * n + YIdx) * ML + 11] = NbrIdx[11];
            }
          }
        }
      }
    }
  }
}

__global__ void cal_Indepl13(double* C, int* G, int* GPrime, int* mutex, int* Sepset, double* pMax, double th, int n,
                             int num_heterogeneous_rows) {
  int YIdx;
  int XIdx = blockIdx.y;
  int NbrIdxPointer[13];
  int SizeOfArr;
  int NumberOfJump;
  int NumOfGivenJump;
  int NumOfComb;
  __shared__ int NoEdgeFlag;
  int NbrIdx[13];

  double M0;
  double M1[2][13];
  double M2[13][13];
  double M2Inv[13][13];
  double M1MulM2Inv[2][13];
  double H[2][2];
  double rho;
  double Z;
  // pseudo-inverse parameter
  double v[13][13];
  double w[13], rv1[13];
  double res1[13][13];

  extern __shared__ int G_Chunk[];

  NoEdgeFlag = 0;
  SizeOfArr = GPrime[XIdx * n + n - 1];
  if (SizeOfArr <= 13) {
    return;
  }

  if ((SizeOfArr % ParGivenL13) == 0) {
    NumberOfJump = SizeOfArr / ParGivenL13;
  } else {
    NumberOfJump = SizeOfArr / ParGivenL13 + 1;
  }
  // Copy Row Xid from GPrime to G_chunck
  for (int cnt = 0; cnt < NumberOfJump; cnt++) {
    if ((threadIdx.x + cnt * ParGivenL13) < SizeOfArr) {
      G_Chunk[threadIdx.x + cnt * ParGivenL13] = GPrime[XIdx * n + threadIdx.x + cnt * ParGivenL13];
    }
    __syncthreads();
  }

  BINOM(SizeOfArr, 13, &NumOfComb);
  if ((NumOfComb % (ParGivenL13 * NumOfBlockForEachNodeL13)) == 0) {
    NumOfGivenJump = NumOfComb / (ParGivenL13 * NumOfBlockForEachNodeL13);
  } else {
    NumOfGivenJump = NumOfComb / (ParGivenL13 * NumOfBlockForEachNodeL13) + 1;
  }

  for (int d1 = 0; d1 < NumOfGivenJump; d1++) {
    __syncthreads();
    if (NoEdgeFlag == 1) {
      return;
    }
    if (threadIdx.x + blockIdx.x * ParGivenL13 + d1 * ParGivenL13 * NumOfBlockForEachNodeL13 < NumOfComb) {
      __syncthreads();
      NoEdgeFlag = 1;
      __syncthreads();
      IthCombination(NbrIdxPointer, SizeOfArr, 13,
                     threadIdx.x + blockIdx.x * ParGivenL13 + d1 * ParGivenL13 * NumOfBlockForEachNodeL13 + 1);
      for (int tmp = 0; tmp < 13; tmp++) {
        NbrIdx[tmp] = G_Chunk[NbrIdxPointer[tmp] - 1];
      }

      for (int c1 = 0; c1 < 13; c1++) {
        for (int c2 = 0; c2 < 13; c2++) {
          if (c1 > c2) {
            M2[c1][c2] = M2[c2][c1];
          } else if (c1 == c2) {
            M2[c1][c1] = 1;
          } else {
            M2[c1][c2] = C[NbrIdx[c1] * n + NbrIdx[c2]];
          }
        }
      }

      for (int c1 = 0; c1 < 13; c1++) {
        M1[0][c1] = C[XIdx * n + NbrIdx[c1]];
      }

      pseudoinversel13(M2, M2Inv, v, rv1, w, res1);
      for (int d2 = 0; d2 < SizeOfArr; d2++) {
        if ((d2 == (NbrIdxPointer[0] - 1)) || (d2 == (NbrIdxPointer[1] - 1)) || (d2 == (NbrIdxPointer[2] - 1)) ||
            (d2 == (NbrIdxPointer[3] - 1)) || (d2 == (NbrIdxPointer[4] - 1)) || (d2 == (NbrIdxPointer[5] - 1)) ||
            (d2 == (NbrIdxPointer[6] - 1)) || (d2 == (NbrIdxPointer[7] - 1)) || (d2 == (NbrIdxPointer[8] - 1)) ||
            (d2 == (NbrIdxPointer[9] - 1)) || (d2 == (NbrIdxPointer[10] - 1)) || (d2 == (NbrIdxPointer[11] - 1)) ||
            (d2 == (NbrIdxPointer[12] - 1))) {
          continue;
        }
        YIdx = G_Chunk[d2];

        if (XIdx < num_heterogeneous_rows || YIdx < num_heterogeneous_rows) {
          continue;
        }

        if (G[XIdx * n + YIdx] == 1) {
          NoEdgeFlag = 0;
          M0 = C[XIdx * n + YIdx];
          // Beginning Of the Indep Test Calculation
          for (int c1 = 0; c1 < 13; c1++) {
            M1[1][c1] = C[YIdx * n + NbrIdx[c1]];
          }
          // Begin to calculate I2Inv Using pseudo-inverse
          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 13; c2++) {
              M1MulM2Inv[c1][c2] = 0;
              for (int c3 = 0; c3 < 13; c3++) M1MulM2Inv[c1][c2] += M1[c1][c3] * M2Inv[c3][c2];
            }
          }

          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 2; c2++) {
              H[c1][c2] = 0;
              for (int c3 = 0; c3 < 13; c3++) H[c1][c2] += M1MulM2Inv[c1][c3] * M1[c2][c3];
            }
          }
          H[0][0] = 1 - H[0][0];
          H[0][1] = M0 - H[0][1];
          H[1][1] = 1 - H[1][1];

          rho = H[0][1] / (sqrt(abs(H[0][0] * H[1][1])));
          Z = abs(0.5 * log(abs((1 + rho) / (1 - rho))));

          if (Z < th) {
            if (atomicCAS(&mutex[XIdx * n + YIdx], 0, 1) == 0) {  // lock
              G[XIdx * n + YIdx] = 0;
              G[YIdx * n + XIdx] = 0;
              pMax[XIdx * n + YIdx] = Z;
              Sepset[(XIdx * n + YIdx) * ML] = NbrIdx[0];
              Sepset[(XIdx * n + YIdx) * ML + 1] = NbrIdx[1];
              Sepset[(XIdx * n + YIdx) * ML + 2] = NbrIdx[2];
              Sepset[(XIdx * n + YIdx) * ML + 3] = NbrIdx[3];
              Sepset[(XIdx * n + YIdx) * ML + 4] = NbrIdx[4];
              Sepset[(XIdx * n + YIdx) * ML + 5] = NbrIdx[5];
              Sepset[(XIdx * n + YIdx) * ML + 6] = NbrIdx[6];
              Sepset[(XIdx * n + YIdx) * ML + 7] = NbrIdx[7];
              Sepset[(XIdx * n + YIdx) * ML + 8] = NbrIdx[8];
              Sepset[(XIdx * n + YIdx) * ML + 9] = NbrIdx[9];
              Sepset[(XIdx * n + YIdx) * ML + 10] = NbrIdx[10];
              Sepset[(XIdx * n + YIdx) * ML + 11] = NbrIdx[11];
              Sepset[(XIdx * n + YIdx) * ML + 12] = NbrIdx[12];
            }
          }
        }
      }
    }
  }
}

__global__ void cal_Indepl14(double* C, int* G, int* GPrime, int* mutex, int* Sepset, double* pMax, double th, int n,
                             int num_heterogeneous_rows) {
  int YIdx;
  int XIdx = blockIdx.y;
  int NbrIdxPointer[14];
  int SizeOfArr;
  int NumberOfJump;
  int NumOfGivenJump;
  int NumOfComb;
  __shared__ int NoEdgeFlag;
  int NbrIdx[14];

  double M0;
  double M1[2][14];
  double M2[14][14];
  double M2Inv[14][14];
  double M1MulM2Inv[2][14];
  double H[2][2];
  double rho;
  double Z;
  // pseudo-inverse parameter
  double v[14][14];
  double w[14], rv1[14];
  double res1[14][14];

  extern __shared__ int G_Chunk[];

  NoEdgeFlag = 0;
  SizeOfArr = GPrime[XIdx * n + n - 1];
  if (SizeOfArr <= 14) {
    return;
  }

  if ((SizeOfArr % ParGivenL14) == 0) {
    NumberOfJump = SizeOfArr / ParGivenL14;
  } else {
    NumberOfJump = SizeOfArr / ParGivenL14 + 1;
  }
  // Copy Row Xid from GPrime to G_chunck
  for (int cnt = 0; cnt < NumberOfJump; cnt++) {
    if ((threadIdx.x + cnt * ParGivenL14) < SizeOfArr) {
      G_Chunk[threadIdx.x + cnt * ParGivenL14] = GPrime[XIdx * n + threadIdx.x + cnt * ParGivenL14];
    }
    __syncthreads();
  }

  BINOM(SizeOfArr, 14, &NumOfComb);
  if ((NumOfComb % (ParGivenL14 * NumOfBlockForEachNodeL14)) == 0) {
    NumOfGivenJump = NumOfComb / (ParGivenL14 * NumOfBlockForEachNodeL14);
  } else {
    NumOfGivenJump = NumOfComb / (ParGivenL14 * NumOfBlockForEachNodeL14) + 1;
  }

  for (int d1 = 0; d1 < NumOfGivenJump; d1++) {
    __syncthreads();
    if (NoEdgeFlag == 1) {
      return;
    }
    if (threadIdx.x + blockIdx.x * ParGivenL14 + d1 * ParGivenL14 * NumOfBlockForEachNodeL14 < NumOfComb) {
      __syncthreads();
      NoEdgeFlag = 1;
      __syncthreads();
      IthCombination(NbrIdxPointer, SizeOfArr, 14,
                     threadIdx.x + blockIdx.x * ParGivenL14 + d1 * ParGivenL14 * NumOfBlockForEachNodeL14 + 1);
      for (int tmp = 0; tmp < 14; tmp++) {
        NbrIdx[tmp] = G_Chunk[NbrIdxPointer[tmp] - 1];
      }

      for (int c1 = 0; c1 < 14; c1++) {
        for (int c2 = 0; c2 < 14; c2++) {
          if (c1 > c2) {
            M2[c1][c2] = M2[c2][c1];
          } else if (c1 == c2) {
            M2[c1][c1] = 1;
          } else {
            M2[c1][c2] = C[NbrIdx[c1] * n + NbrIdx[c2]];
          }
        }
      }

      for (int c1 = 0; c1 < 14; c1++) {
        M1[0][c1] = C[XIdx * n + NbrIdx[c1]];
      }

      pseudoinversel14(M2, M2Inv, v, rv1, w, res1);
      for (int d2 = 0; d2 < SizeOfArr; d2++) {
        if ((d2 == (NbrIdxPointer[0] - 1)) || (d2 == (NbrIdxPointer[1] - 1)) || (d2 == (NbrIdxPointer[2] - 1)) ||
            (d2 == (NbrIdxPointer[3] - 1)) || (d2 == (NbrIdxPointer[4] - 1)) || (d2 == (NbrIdxPointer[5] - 1)) ||
            (d2 == (NbrIdxPointer[6] - 1)) || (d2 == (NbrIdxPointer[7] - 1)) || (d2 == (NbrIdxPointer[8] - 1)) ||
            (d2 == (NbrIdxPointer[9] - 1)) || (d2 == (NbrIdxPointer[10] - 1)) || (d2 == (NbrIdxPointer[11] - 1)) ||
            (d2 == (NbrIdxPointer[12] - 1)) || (d2 == (NbrIdxPointer[13] - 1))) {
          continue;
        }
        YIdx = G_Chunk[d2];

        if (XIdx < num_heterogeneous_rows || YIdx < num_heterogeneous_rows) {
          continue;
        }

        if (G[XIdx * n + YIdx] == 1) {
          NoEdgeFlag = 0;
          M0 = C[XIdx * n + YIdx];
          // Beginning Of the Indep Test Calculation
          for (int c1 = 0; c1 < 14; c1++) {
            M1[1][c1] = C[YIdx * n + NbrIdx[c1]];
          }
          // Begin to calculate I2Inv Using pseudo-inverse
          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 14; c2++) {
              M1MulM2Inv[c1][c2] = 0;
              for (int c3 = 0; c3 < 14; c3++) M1MulM2Inv[c1][c2] += M1[c1][c3] * M2Inv[c3][c2];
            }
          }

          for (int c1 = 0; c1 < 2; c1++) {
            for (int c2 = 0; c2 < 2; c2++) {
              H[c1][c2] = 0;
              for (int c3 = 0; c3 < 14; c3++) H[c1][c2] += M1MulM2Inv[c1][c3] * M1[c2][c3];
            }
          }
          H[0][0] = 1 - H[0][0];
          H[0][1] = M0 - H[0][1];
          H[1][1] = 1 - H[1][1];

          rho = H[0][1] / (sqrt(abs(H[0][0] * H[1][1])));
          Z = abs(0.5 * log(abs((1 + rho) / (1 - rho))));

          if (Z < th) {
            if (atomicCAS(&mutex[XIdx * n + YIdx], 0, 1) == 0) {  // lock
              G[XIdx * n + YIdx] = 0;
              G[YIdx * n + XIdx] = 0;
              pMax[XIdx * n + YIdx] = Z;
              Sepset[(XIdx * n + YIdx) * ML] = NbrIdx[0];
              Sepset[(XIdx * n + YIdx) * ML + 1] = NbrIdx[1];
              Sepset[(XIdx * n + YIdx) * ML + 2] = NbrIdx[2];
              Sepset[(XIdx * n + YIdx) * ML + 3] = NbrIdx[3];
              Sepset[(XIdx * n + YIdx) * ML + 4] = NbrIdx[4];
              Sepset[(XIdx * n + YIdx) * ML + 5] = NbrIdx[5];
              Sepset[(XIdx * n + YIdx) * ML + 6] = NbrIdx[6];
              Sepset[(XIdx * n + YIdx) * ML + 7] = NbrIdx[7];
              Sepset[(XIdx * n + YIdx) * ML + 8] = NbrIdx[8];
              Sepset[(XIdx * n + YIdx) * ML + 9] = NbrIdx[9];
              Sepset[(XIdx * n + YIdx) * ML + 10] = NbrIdx[10];
              Sepset[(XIdx * n + YIdx) * ML + 11] = NbrIdx[11];
              Sepset[(XIdx * n + YIdx) * ML + 12] = NbrIdx[12];
              Sepset[(XIdx * n + YIdx) * ML + 13] = NbrIdx[13];
            }
          }
        }
      }
    }
  }
}

__global__ void Scan(int* G_ScanOut, int* G_ScanIn, int Step, int GSize) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int row = blockIdx.y;
  if ((index < Step) && (index < GSize)) {
    G_ScanOut[row * GSize + index] = G_ScanIn[row * GSize + index];
  }
  if ((index >= Step) && (index < GSize)) {
    G_ScanOut[row * GSize + index] = G_ScanIn[row * GSize + index] + G_ScanIn[row * GSize + index - Step];
  }
}

__global__ void Compact(int* G_Compact, const int* G, const int* G_ScanRes, int GSize) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int row = blockIdx.y;
  int CompactIdx;
  if (index < GSize) {
    if ((G[row * GSize + index] == 1)) {
      CompactIdx = G_ScanRes[row * GSize + index] - 1;
      G_Compact[row * GSize + CompactIdx] = index;
    }
    if (index >= G_ScanRes[row * GSize + GSize - 1]) {
      if (index != (GSize - 1)) {
        G_Compact[row * GSize + index] = 0;
      } else {
        G_Compact[row * GSize + GSize - 1] = G_ScanRes[row * GSize + GSize - 1];
      }
    }
  }
}

__device__ double PYTHAG(double a, double b) {
  double at = fabs(a), bt = fabs(b), ct, result;

  if (at > bt) {
    ct = bt / at;
    result = at * sqrt(1.0 + ct * ct);
  } else if (bt > 0.0) {
    ct = at / bt;
    result = bt * sqrt(1.0 + ct * ct);
  } else {
    result = 0.0;
  }
  return (result);
}

__device__ void pseudoinversel2(double M2[][2], double M2Inv[][2]) {
  double A[2][2];
  double M[2][2];
  double L[2][2];
  double newL[2][2];
  double temp[2][2];
  double temp1[2][2];
  double temp2[2][2];
  double temp3[2][2];

  double tol = 999.99;
  double aux = 0.0;
  double det = 0.0;

  int r = 0;
  int size = 2;

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      A[i][j] = 0.0;
      M[i][j] = 0.0;
      L[i][j] = 0.0;
      newL[i][j] = 0.0;
      temp[i][j] = 0.0;
      temp1[i][j] = 0.0;
      temp2[i][j] = 0.0;
      temp3[i][j] = 0.0;
      M2Inv[i][j] = 0.0;
    }
  }

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      for (int k = 0; k < size; k++) {
        A[i][j] += M2[i][k] * M2[k][j];
      }
    }
  }

  for (int i = 0; i < size; i++) {
    if (tol > A[i][i] && A[i][i] > 0) {
      tol = A[i][i];
    }
  }

  // tol = tol * 1e-9 accroding to paper
  tol = tol * (1e-20);

  for (int k = 0; k < size; k++) {
    if (r == 0) {
      for (int i = k; i < size; i++) {
        L[i][r] = A[i][k];
      }
    } else {
      for (int i = k; i < size; ++i) {
        for (int l = 0; l < r; l++) {
          temp[i][k] += L[i][l] * L[k][l];
        }
      }
      for (int i = k; i < size; i++) {
        L[i][r] = A[i][k] - temp[i][k];
      }
    }
    // check with threshold
    if (L[k][r] > tol) {
      L[k][r] = sqrt(L[k][r]);
      if (k < size) {
        for (int i = k + 1; i < size; i++) {
          L[i][r] = L[i][r] / L[k][r];
        }
      }
    } else {
      r--;
    }
    r++;
  }

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < r; j++) {
      newL[i][j] = L[i][j];
    }
  }

  for (int i = 0; i < r; i++) {
    for (int j = 0; j < r; j++) {
      for (int k = 0; k < size; k++) {
        M[i][j] += newL[k][i] * newL[k][j];
      }
    }
  }

  /*
   * it's time to compute inv(M) in this stage M is 2*2 so
   * I use close form of 2*2
   */
  if (r == 1) {
    M[0][0] = 1 / M[0][0];
  } else if (r == 2) {
    aux = 0.0;
    det = 1 / (M[0][0] * M[1][1] - M[0][1] * M[1][0]);
    aux = M[0][0];
    M[0][0] = det * M[1][1];
    M[1][1] = det * aux;
    M[0][1] = (-1 * det) * M[0][1];
    M[1][0] = (-1 * det) * M[1][0];
  }

  /*At the final step we must compute L * M * M * L' * G'
   * at first I compute   temp1 = L  * M
   * after that I compute temp2 = L' * G'
   * after that I compute temp3 = M  * temp2
   * finally I compute   output = temp1 * temp3
   */

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < r; j++) {
      for (int k = 0; k < r; k++) {
        temp1[i][j] += newL[i][k] * M[k][j];
      }
    }
  }

  for (int i = 0; i < r; i++) {
    for (int j = 0; j < size; j++) {
      for (int k = 0; k < size; k++) {
        temp2[i][j] += newL[k][i] * M2[k][j];
      }
    }
  }

  for (int i = 0; i < r; i++) {
    for (int j = 0; j < size; j++) {
      for (int k = 0; k < size; k++) {
        temp3[i][j] += M[i][k] * temp2[k][j];
      }
    }
  }

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      for (int k = 0; k < size; k++) {
        M2Inv[i][j] += temp1[i][k] * temp3[k][j];
      }
    }
  }
}

__device__ void pseudoinversel3(double M2[][3], double M2Inv[][3]) {
  double A[3][3];
  double M[3][3];
  double tempM[3][3];
  double L[3][3];
  double newL[3][3];
  double temp[3][3];
  double temp1[3][3];
  double temp2[3][3];
  double temp3[3][3];

  double tol = 999.99;
  double det = 0.0;

  int r = 0;
  int size = 3;

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      A[i][j] = 0.0;
      M[i][j] = 0.0;
      L[i][j] = 0.0;
      newL[i][j] = 0.0;
      temp[i][j] = 0.0;
      temp1[i][j] = 0.0;
      temp2[i][j] = 0.0;
      temp3[i][j] = 0.0;
      M2Inv[i][j] = 0.0;
      tempM[i][j] = 0.0;
    }
  }

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      for (int k = 0; k < size; k++) {
        A[i][j] += M2[i][k] * M2[k][j];
      }
    }
  }

  for (int i = 0; i < size; i++) {
    if (tol > A[i][i] && A[i][i] > 0) {
      tol = A[i][i];
    }
  }

  // tol = tol * 1e-9 accroding to paper
  tol = tol * (1e-20);

  for (int k = 0; k < size; k++) {
    if (r == 0) {
      for (int i = k; i < size; i++) {
        L[i][r] = A[i][k];
      }
    } else {
      for (int i = k; i < size; ++i) {
        for (int l = 0; l < r; l++) {
          temp[i][k] += L[i][l] * L[k][l];
        }
      }
      for (int i = k; i < size; i++) {
        L[i][r] = A[i][k] - temp[i][k];
      }
    }
    // check with threshold
    if (L[k][r] > tol) {
      L[k][r] = sqrt(L[k][r]);
      if (k < size) {
        for (int i = k + 1; i < size; i++) {
          L[i][r] = L[i][r] / L[k][r];
        }
      }
    } else {
      r--;
    }
    r++;
  }

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < r; j++) {
      newL[i][j] = L[i][j];
    }
  }

  for (int i = 0; i < r; i++) {
    for (int j = 0; j < r; j++) {
      for (int k = 0; k < size; k++) {
        tempM[i][j] += newL[k][i] * newL[k][j];
      }
    }
  }

  /*
   * it's time to compute inv(M) in this stage M is 2*2 so
   * I use close form of 2*2
   */
  if (r == 1) {
    M[0][0] = 1 / tempM[0][0];
  } else if (r == 2) {
    det = 1 / (tempM[0][0] * tempM[1][1] - tempM[0][1] * tempM[1][0]);
    M[0][0] = det * tempM[1][1];
    M[1][1] = det * tempM[0][0];
    M[0][1] = (-1 * det) * tempM[0][1];
    M[1][0] = (-1 * det) * tempM[1][0];
  } else {
    inverse(tempM, M);
  }

  /*At the final step we must compute L * M * M * L' * G'
   * at first I compute   temp1 = L  * M
   * after that I compute temp2 = L' * G'
   * after that I compute temp3 = M  * temp2
   * finally I compute   output = temp1 * temp3
   */

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < r; j++) {
      for (int k = 0; k < r; k++) {
        temp1[i][j] += newL[i][k] * M[k][j];
      }
    }
  }

  for (int i = 0; i < r; i++) {
    for (int j = 0; j < size; j++) {
      for (int k = 0; k < size; k++) {
        temp2[i][j] += newL[k][i] * M2[k][j];
      }
    }
  }

  for (int i = 0; i < r; i++) {
    for (int j = 0; j < size; j++) {
      for (int k = 0; k < size; k++) {
        temp3[i][j] += M[i][k] * temp2[k][j];
      }
    }
  }

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      for (int k = 0; k < size; k++) {
        M2Inv[i][j] += temp1[i][k] * temp3[k][j];
      }
    }
  }
}

__device__ void pseudoinversel4(double M2[][4], double M2Inv[][4], double v[][4], double* rv1, double* w,
                                double res1[][4]) {
  int m = 4;
  int flag, its, i, j, jj, k, l, nm;
  double c, f, h, s, x, y, z;
  double anorm = 0.0, g = 0.0, scale = 0.0;
  /* Householder reduction to bidiagonal form */
  for (i = 0; i < m; i++) {
    /* left-hand reduction */
    l = i + 1;
    rv1[i] = scale * g;
    g = s = scale = 0.0;
    if (i < m) {
      for (k = i; k < m; k++) scale += fabs(M2[k][i]);
      if (scale) {
        for (k = i; k < m; k++) {
          M2[k][i] = (M2[k][i] / scale);
          s += (M2[k][i] * M2[k][i]);
        }
        f = M2[i][i];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][i] = f - g;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = i; k < m; k++) s += (M2[k][i] * M2[k][j]);
            f = s / h;
            for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
          }
        }
        for (k = i; k < m; k++) M2[k][i] = (M2[k][i] * scale);
      }
    }
    w[i] = scale * g;

    /* right-hand reduction */
    g = s = scale = 0.0;
    if (i < m && i != m - 1) {
      for (k = l; k < m; k++) scale += fabs(M2[i][k]);
      if (scale) {
        for (k = l; k < m; k++) {
          M2[i][k] = (M2[i][k] / scale);
          s += (M2[i][k] * M2[i][k]);
        }
        f = M2[i][l];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][l] = f - g;
        for (k = l; k < m; k++) rv1[k] = M2[i][k] / h;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = l; k < m; k++) s += (M2[j][k] * M2[i][k]);
            for (k = l; k < m; k++) M2[j][k] += (s * rv1[k]);
          }
        }
        for (k = l; k < m; k++) M2[i][k] = M2[i][k] * scale;
      }
    }
    anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
  }

  /* accumulate the right-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    if (i < m - 1) {
      if (g) {
        for (j = l; j < m; j++) v[j][i] = (M2[i][j] / M2[i][l]) / g;
        /* double division to avoid underflow */
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[i][k] * v[k][j]);
          for (k = l; k < m; k++) v[k][j] += (s * v[k][i]);
        }
      }
      for (j = l; j < m; j++) v[i][j] = v[j][i] = 0.0;
    }
    v[i][i] = 1.0;
    g = rv1[i];
    l = i;
  }

  /* accumulate the left-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    l = i + 1;
    g = w[i];
    if (i < m - 1)
      for (j = l; j < m; j++) M2[i][j] = 0.0;
    if (g) {
      g = 1.0 / g;
      if (i != m - 1) {
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[k][i] * M2[k][j]);
          f = (s / M2[i][i]) * g;
          for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
        }
      }
      for (j = i; j < m; j++) M2[j][i] = (M2[j][i] * g);
    } else {
      for (j = i; j < m; j++) M2[j][i] = 0.0;
    }
    ++M2[i][i];
  }

  /* diagonalize the bidiagonal form */
  for (k = m - 1; k >= 0; k--) {     /* loop over singular values */
    for (its = 0; its < 30; its++) { /* loop over allowed iterations */
      flag = 1;
      for (l = k; l >= 0; l--) { /* test for splitting */
        nm = l - 1;
        if (fabs(rv1[l]) + anorm == anorm) {
          flag = 0;
          break;
        }
        if (fabs(w[nm]) + anorm == anorm) break;
      }
      if (flag) {
        c = 0.0;
        s = 1.0;
        for (i = l; i <= k; i++) {
          f = s * rv1[i];
          if (fabs(f) + anorm != anorm) {
            g = w[i];
            h = PYTHAG(f, g);
            w[i] = h;
            h = 1.0 / h;
            c = g * h;
            s = (-f * h);
            for (j = 0; j < m; j++) {
              y = M2[j][nm];
              z = M2[j][i];
              M2[j][nm] = (y * c + z * s);
              M2[j][i] = (z * c - y * s);
            }
          }
        }
      }
      z = w[k];
      if (l == k) {    /* convergence */
        if (z < 0.0) { /* make singular value nonnegative */
          w[k] = (-z);
          for (j = 0; j < m; j++) v[j][k] = (-v[j][k]);
        }
        break;
      }
      if (its >= 30) {
        printf("Not converged\n");
      }

      /* shift from bottom 2 x 2 minor */
      x = w[l];
      nm = k - 1;
      y = w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = PYTHAG(f, 1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

      /* next QR transformation */
      c = s = 1.0;
      for (j = l; j <= nm; j++) {
        i = j + 1;
        g = rv1[i];
        y = w[i];
        h = s * g;
        g = c * g;
        z = PYTHAG(f, h);
        rv1[j] = z;
        c = f / z;
        s = h / z;
        f = x * c + g * s;
        g = g * c - x * s;
        h = y * s;
        y = y * c;
        for (jj = 0; jj < m; jj++) {
          x = v[jj][j];
          z = v[jj][i];
          v[jj][j] = (x * c + z * s);
          v[jj][i] = (z * c - x * s);
        }
        z = PYTHAG(f, h);
        w[j] = z;
        if (z) {
          z = 1.0 / z;
          c = f * z;
          s = h * z;
        }
        f = (c * g) + (s * y);
        x = (c * y) - (s * g);
        for (jj = 0; jj < m; jj++) {
          y = M2[jj][j];
          z = M2[jj][i];
          M2[jj][j] = (y * c + z * s);
          M2[jj][i] = (z * c - y * s);
        }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      w[k] = x;
    }
  }

  // start compute inverse matrix

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      res1[rowNumber][colNumber] = v[rowNumber][colNumber] / w[colNumber];
    }
  }

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      M2Inv[rowNumber][colNumber] = 0;
      for (int thirdIndex = 0; thirdIndex < m; thirdIndex++) {
        M2Inv[rowNumber][colNumber] =
            M2Inv[rowNumber][colNumber] + res1[rowNumber][thirdIndex] * M2[colNumber][thirdIndex];
      }
    }
  }
}

__device__ void pseudoinversel5(double M2[][5], double M2Inv[][5], double v[][5], double* rv1, double* w,
                                double res1[][5]) {
  int m = 5;
  int flag, its, i, j, jj, k, l, nm;
  double c, f, h, s, x, y, z;
  double anorm = 0.0, g = 0.0, scale = 0.0;
  /* Householder reduction to bidiagonal form */
  for (i = 0; i < m; i++) {
    /* left-hand reduction */
    l = i + 1;
    rv1[i] = scale * g;
    g = s = scale = 0.0;
    if (i < m) {
      for (k = i; k < m; k++) scale += fabs(M2[k][i]);
      if (scale) {
        for (k = i; k < m; k++) {
          M2[k][i] = (M2[k][i] / scale);
          s += (M2[k][i] * M2[k][i]);
        }
        f = M2[i][i];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][i] = f - g;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = i; k < m; k++) s += (M2[k][i] * M2[k][j]);
            f = s / h;
            for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
          }
        }
        for (k = i; k < m; k++) M2[k][i] = (M2[k][i] * scale);
      }
    }
    w[i] = scale * g;

    /* right-hand reduction */
    g = s = scale = 0.0;
    if (i < m && i != m - 1) {
      for (k = l; k < m; k++) scale += fabs(M2[i][k]);
      if (scale) {
        for (k = l; k < m; k++) {
          M2[i][k] = (M2[i][k] / scale);
          s += (M2[i][k] * M2[i][k]);
        }
        f = M2[i][l];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][l] = f - g;
        for (k = l; k < m; k++) rv1[k] = M2[i][k] / h;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = l; k < m; k++) s += (M2[j][k] * M2[i][k]);
            for (k = l; k < m; k++) M2[j][k] += (s * rv1[k]);
          }
        }
        for (k = l; k < m; k++) M2[i][k] = M2[i][k] * scale;
      }
    }
    anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
  }

  /* accumulate the right-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    if (i < m - 1) {
      if (g) {
        for (j = l; j < m; j++) v[j][i] = (M2[i][j] / M2[i][l]) / g;
        /* double division to avoid underflow */
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[i][k] * v[k][j]);
          for (k = l; k < m; k++) v[k][j] += (s * v[k][i]);
        }
      }
      for (j = l; j < m; j++) v[i][j] = v[j][i] = 0.0;
    }
    v[i][i] = 1.0;
    g = rv1[i];
    l = i;
  }

  /* accumulate the left-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    l = i + 1;
    g = w[i];
    if (i < m - 1)
      for (j = l; j < m; j++) M2[i][j] = 0.0;
    if (g) {
      g = 1.0 / g;
      if (i != m - 1) {
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[k][i] * M2[k][j]);
          f = (s / M2[i][i]) * g;
          for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
        }
      }
      for (j = i; j < m; j++) M2[j][i] = (M2[j][i] * g);
    } else {
      for (j = i; j < m; j++) M2[j][i] = 0.0;
    }
    ++M2[i][i];
  }

  /* diagonalize the bidiagonal form */
  for (k = m - 1; k >= 0; k--) {     /* loop over singular values */
    for (its = 0; its < 30; its++) { /* loop over allowed iterations */
      flag = 1;
      for (l = k; l >= 0; l--) { /* test for splitting */
        nm = l - 1;
        if (fabs(rv1[l]) + anorm == anorm) {
          flag = 0;
          break;
        }
        if (fabs(w[nm]) + anorm == anorm) break;
      }
      if (flag) {
        c = 0.0;
        s = 1.0;
        for (i = l; i <= k; i++) {
          f = s * rv1[i];
          if (fabs(f) + anorm != anorm) {
            g = w[i];
            h = PYTHAG(f, g);
            w[i] = h;
            h = 1.0 / h;
            c = g * h;
            s = (-f * h);
            for (j = 0; j < m; j++) {
              y = M2[j][nm];
              z = M2[j][i];
              M2[j][nm] = (y * c + z * s);
              M2[j][i] = (z * c - y * s);
            }
          }
        }
      }
      z = w[k];
      if (l == k) {    /* convergence */
        if (z < 0.0) { /* make singular value nonnegative */
          w[k] = (-z);
          for (j = 0; j < m; j++) v[j][k] = (-v[j][k]);
        }
        break;
      }
      if (its >= 30) {
        printf("Not converged\n");
      }

      /* shift from bottom 2 x 2 minor */
      x = w[l];
      nm = k - 1;
      y = w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = PYTHAG(f, 1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

      /* next QR transformation */
      c = s = 1.0;
      for (j = l; j <= nm; j++) {
        i = j + 1;
        g = rv1[i];
        y = w[i];
        h = s * g;
        g = c * g;
        z = PYTHAG(f, h);
        rv1[j] = z;
        c = f / z;
        s = h / z;
        f = x * c + g * s;
        g = g * c - x * s;
        h = y * s;
        y = y * c;
        for (jj = 0; jj < m; jj++) {
          x = v[jj][j];
          z = v[jj][i];
          v[jj][j] = (x * c + z * s);
          v[jj][i] = (z * c - x * s);
        }
        z = PYTHAG(f, h);
        w[j] = z;
        if (z) {
          z = 1.0 / z;
          c = f * z;
          s = h * z;
        }
        f = (c * g) + (s * y);
        x = (c * y) - (s * g);
        for (jj = 0; jj < m; jj++) {
          y = M2[jj][j];
          z = M2[jj][i];
          M2[jj][j] = (y * c + z * s);
          M2[jj][i] = (z * c - y * s);
        }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      w[k] = x;
    }
  }

  // start compute inverse matrix

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      res1[rowNumber][colNumber] = v[rowNumber][colNumber] / w[colNumber];
    }
  }

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      M2Inv[rowNumber][colNumber] = 0;
      for (int thirdIndex = 0; thirdIndex < m; thirdIndex++) {
        M2Inv[rowNumber][colNumber] =
            M2Inv[rowNumber][colNumber] + res1[rowNumber][thirdIndex] * M2[colNumber][thirdIndex];
      }
    }
  }
}
__device__ void pseudoinversel6(double M2[][6], double M2Inv[][6], double v[][6], double* rv1, double* w,
                                double res1[][6]) {
  int m = 6;
  int flag, its, i, j, jj, k, l, nm;
  double c, f, h, s, x, y, z;
  double anorm = 0.0, g = 0.0, scale = 0.0;
  /* Householder reduction to bidiagonal form */
  for (i = 0; i < m; i++) {
    /* left-hand reduction */
    l = i + 1;
    rv1[i] = scale * g;
    g = s = scale = 0.0;
    if (i < m) {
      for (k = i; k < m; k++) scale += fabs(M2[k][i]);
      if (scale) {
        for (k = i; k < m; k++) {
          M2[k][i] = (M2[k][i] / scale);
          s += (M2[k][i] * M2[k][i]);
        }
        f = M2[i][i];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][i] = f - g;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = i; k < m; k++) s += (M2[k][i] * M2[k][j]);
            f = s / h;
            for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
          }
        }
        for (k = i; k < m; k++) M2[k][i] = (M2[k][i] * scale);
      }
    }
    w[i] = scale * g;

    /* right-hand reduction */
    g = s = scale = 0.0;
    if (i < m && i != m - 1) {
      for (k = l; k < m; k++) scale += fabs(M2[i][k]);
      if (scale) {
        for (k = l; k < m; k++) {
          M2[i][k] = (M2[i][k] / scale);
          s += (M2[i][k] * M2[i][k]);
        }
        f = M2[i][l];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][l] = f - g;
        for (k = l; k < m; k++) rv1[k] = M2[i][k] / h;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = l; k < m; k++) s += (M2[j][k] * M2[i][k]);
            for (k = l; k < m; k++) M2[j][k] += (s * rv1[k]);
          }
        }
        for (k = l; k < m; k++) M2[i][k] = M2[i][k] * scale;
      }
    }
    anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
  }

  /* accumulate the right-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    if (i < m - 1) {
      if (g) {
        for (j = l; j < m; j++) v[j][i] = (M2[i][j] / M2[i][l]) / g;
        /* double division to avoid underflow */
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[i][k] * v[k][j]);
          for (k = l; k < m; k++) v[k][j] += (s * v[k][i]);
        }
      }
      for (j = l; j < m; j++) v[i][j] = v[j][i] = 0.0;
    }
    v[i][i] = 1.0;
    g = rv1[i];
    l = i;
  }

  /* accumulate the left-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    l = i + 1;
    g = w[i];
    if (i < m - 1)
      for (j = l; j < m; j++) M2[i][j] = 0.0;
    if (g) {
      g = 1.0 / g;
      if (i != m - 1) {
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[k][i] * M2[k][j]);
          f = (s / M2[i][i]) * g;
          for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
        }
      }
      for (j = i; j < m; j++) M2[j][i] = (M2[j][i] * g);
    } else {
      for (j = i; j < m; j++) M2[j][i] = 0.0;
    }
    ++M2[i][i];
  }

  /* diagonalize the bidiagonal form */
  for (k = m - 1; k >= 0; k--) {     /* loop over singular values */
    for (its = 0; its < 30; its++) { /* loop over allowed iterations */
      flag = 1;
      for (l = k; l >= 0; l--) { /* test for splitting */
        nm = l - 1;
        if (fabs(rv1[l]) + anorm == anorm) {
          flag = 0;
          break;
        }
        if (fabs(w[nm]) + anorm == anorm) break;
      }
      if (flag) {
        c = 0.0;
        s = 1.0;
        for (i = l; i <= k; i++) {
          f = s * rv1[i];
          if (fabs(f) + anorm != anorm) {
            g = w[i];
            h = PYTHAG(f, g);
            w[i] = h;
            h = 1.0 / h;
            c = g * h;
            s = (-f * h);
            for (j = 0; j < m; j++) {
              y = M2[j][nm];
              z = M2[j][i];
              M2[j][nm] = (y * c + z * s);
              M2[j][i] = (z * c - y * s);
            }
          }
        }
      }
      z = w[k];
      if (l == k) {    /* convergence */
        if (z < 0.0) { /* make singular value nonnegative */
          w[k] = (-z);
          for (j = 0; j < m; j++) v[j][k] = (-v[j][k]);
        }
        break;
      }
      if (its >= 30) {
        printf("Not converged\n");
      }

      /* shift from bottom 2 x 2 minor */
      x = w[l];
      nm = k - 1;
      y = w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = PYTHAG(f, 1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

      /* next QR transformation */
      c = s = 1.0;
      for (j = l; j <= nm; j++) {
        i = j + 1;
        g = rv1[i];
        y = w[i];
        h = s * g;
        g = c * g;
        z = PYTHAG(f, h);
        rv1[j] = z;
        c = f / z;
        s = h / z;
        f = x * c + g * s;
        g = g * c - x * s;
        h = y * s;
        y = y * c;
        for (jj = 0; jj < m; jj++) {
          x = v[jj][j];
          z = v[jj][i];
          v[jj][j] = (x * c + z * s);
          v[jj][i] = (z * c - x * s);
        }
        z = PYTHAG(f, h);
        w[j] = z;
        if (z) {
          z = 1.0 / z;
          c = f * z;
          s = h * z;
        }
        f = (c * g) + (s * y);
        x = (c * y) - (s * g);
        for (jj = 0; jj < m; jj++) {
          y = M2[jj][j];
          z = M2[jj][i];
          M2[jj][j] = (y * c + z * s);
          M2[jj][i] = (z * c - y * s);
        }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      w[k] = x;
    }
  }

  // start compute inverse matrix

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      res1[rowNumber][colNumber] = v[rowNumber][colNumber] / w[colNumber];
    }
  }

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      M2Inv[rowNumber][colNumber] = 0;
      for (int thirdIndex = 0; thirdIndex < m; thirdIndex++) {
        M2Inv[rowNumber][colNumber] =
            M2Inv[rowNumber][colNumber] + res1[rowNumber][thirdIndex] * M2[colNumber][thirdIndex];
      }
    }
  }
}

__device__ void pseudoinversel7(double M2[][7], double M2Inv[][7], double v[][7], double* rv1, double* w,
                                double res1[][7]) {
  int m = 7;
  int flag, its, i, j, jj, k, l, nm;
  double c, f, h, s, x, y, z;
  double anorm = 0.0, g = 0.0, scale = 0.0;
  /* Householder reduction to bidiagonal form */
  for (i = 0; i < m; i++) {
    /* left-hand reduction */
    l = i + 1;
    rv1[i] = scale * g;
    g = s = scale = 0.0;
    if (i < m) {
      for (k = i; k < m; k++) scale += fabs(M2[k][i]);
      if (scale) {
        for (k = i; k < m; k++) {
          M2[k][i] = (M2[k][i] / scale);
          s += (M2[k][i] * M2[k][i]);
        }
        f = M2[i][i];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][i] = f - g;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = i; k < m; k++) s += (M2[k][i] * M2[k][j]);
            f = s / h;
            for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
          }
        }
        for (k = i; k < m; k++) M2[k][i] = (M2[k][i] * scale);
      }
    }
    w[i] = scale * g;

    /* right-hand reduction */
    g = s = scale = 0.0;
    if (i < m && i != m - 1) {
      for (k = l; k < m; k++) scale += fabs(M2[i][k]);
      if (scale) {
        for (k = l; k < m; k++) {
          M2[i][k] = (M2[i][k] / scale);
          s += (M2[i][k] * M2[i][k]);
        }
        f = M2[i][l];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][l] = f - g;
        for (k = l; k < m; k++) rv1[k] = M2[i][k] / h;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = l; k < m; k++) s += (M2[j][k] * M2[i][k]);
            for (k = l; k < m; k++) M2[j][k] += (s * rv1[k]);
          }
        }
        for (k = l; k < m; k++) M2[i][k] = M2[i][k] * scale;
      }
    }
    anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
  }

  /* accumulate the right-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    if (i < m - 1) {
      if (g) {
        for (j = l; j < m; j++) v[j][i] = (M2[i][j] / M2[i][l]) / g;
        /* double division to avoid underflow */
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[i][k] * v[k][j]);
          for (k = l; k < m; k++) v[k][j] += (s * v[k][i]);
        }
      }
      for (j = l; j < m; j++) v[i][j] = v[j][i] = 0.0;
    }
    v[i][i] = 1.0;
    g = rv1[i];
    l = i;
  }

  /* accumulate the left-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    l = i + 1;
    g = w[i];
    if (i < m - 1)
      for (j = l; j < m; j++) M2[i][j] = 0.0;
    if (g) {
      g = 1.0 / g;
      if (i != m - 1) {
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[k][i] * M2[k][j]);
          f = (s / M2[i][i]) * g;
          for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
        }
      }
      for (j = i; j < m; j++) M2[j][i] = (M2[j][i] * g);
    } else {
      for (j = i; j < m; j++) M2[j][i] = 0.0;
    }
    ++M2[i][i];
  }

  /* diagonalize the bidiagonal form */
  for (k = m - 1; k >= 0; k--) {     /* loop over singular values */
    for (its = 0; its < 30; its++) { /* loop over allowed iterations */
      flag = 1;
      for (l = k; l >= 0; l--) { /* test for splitting */
        nm = l - 1;
        if (fabs(rv1[l]) + anorm == anorm) {
          flag = 0;
          break;
        }
        if (fabs(w[nm]) + anorm == anorm) break;
      }
      if (flag) {
        c = 0.0;
        s = 1.0;
        for (i = l; i <= k; i++) {
          f = s * rv1[i];
          if (fabs(f) + anorm != anorm) {
            g = w[i];
            h = PYTHAG(f, g);
            w[i] = h;
            h = 1.0 / h;
            c = g * h;
            s = (-f * h);
            for (j = 0; j < m; j++) {
              y = M2[j][nm];
              z = M2[j][i];
              M2[j][nm] = (y * c + z * s);
              M2[j][i] = (z * c - y * s);
            }
          }
        }
      }
      z = w[k];
      if (l == k) {    /* convergence */
        if (z < 0.0) { /* make singular value nonnegative */
          w[k] = (-z);
          for (j = 0; j < m; j++) v[j][k] = (-v[j][k]);
        }
        break;
      }
      if (its >= 30) {
        printf("Not converged\n");
      }

      /* shift from bottom 2 x 2 minor */
      x = w[l];
      nm = k - 1;
      y = w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = PYTHAG(f, 1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

      /* next QR transformation */
      c = s = 1.0;
      for (j = l; j <= nm; j++) {
        i = j + 1;
        g = rv1[i];
        y = w[i];
        h = s * g;
        g = c * g;
        z = PYTHAG(f, h);
        rv1[j] = z;
        c = f / z;
        s = h / z;
        f = x * c + g * s;
        g = g * c - x * s;
        h = y * s;
        y = y * c;
        for (jj = 0; jj < m; jj++) {
          x = v[jj][j];
          z = v[jj][i];
          v[jj][j] = (x * c + z * s);
          v[jj][i] = (z * c - x * s);
        }
        z = PYTHAG(f, h);
        w[j] = z;
        if (z) {
          z = 1.0 / z;
          c = f * z;
          s = h * z;
        }
        f = (c * g) + (s * y);
        x = (c * y) - (s * g);
        for (jj = 0; jj < m; jj++) {
          y = M2[jj][j];
          z = M2[jj][i];
          M2[jj][j] = (y * c + z * s);
          M2[jj][i] = (z * c - y * s);
        }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      w[k] = x;
    }
  }

  // start compute inverse matrix

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      res1[rowNumber][colNumber] = v[rowNumber][colNumber] / w[colNumber];
    }
  }

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      M2Inv[rowNumber][colNumber] = 0;
      for (int thirdIndex = 0; thirdIndex < m; thirdIndex++) {
        M2Inv[rowNumber][colNumber] =
            M2Inv[rowNumber][colNumber] + res1[rowNumber][thirdIndex] * M2[colNumber][thirdIndex];
      }
    }
  }
}

__device__ void pseudoinversel8(double M2[][8], double M2Inv[][8], double v[][8], double* rv1, double* w,
                                double res1[][8]) {
  int m = 8;
  int flag, its, i, j, jj, k, l, nm;
  double c, f, h, s, x, y, z;
  double anorm = 0.0, g = 0.0, scale = 0.0;
  /* Householder reduction to bidiagonal form */
  for (i = 0; i < m; i++) {
    /* left-hand reduction */
    l = i + 1;
    rv1[i] = scale * g;
    g = s = scale = 0.0;
    if (i < m) {
      for (k = i; k < m; k++) scale += fabs(M2[k][i]);
      if (scale) {
        for (k = i; k < m; k++) {
          M2[k][i] = (M2[k][i] / scale);
          s += (M2[k][i] * M2[k][i]);
        }
        f = M2[i][i];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][i] = f - g;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = i; k < m; k++) s += (M2[k][i] * M2[k][j]);
            f = s / h;
            for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
          }
        }
        for (k = i; k < m; k++) M2[k][i] = (M2[k][i] * scale);
      }
    }
    w[i] = scale * g;

    /* right-hand reduction */
    g = s = scale = 0.0;
    if (i < m && i != m - 1) {
      for (k = l; k < m; k++) scale += fabs(M2[i][k]);
      if (scale) {
        for (k = l; k < m; k++) {
          M2[i][k] = (M2[i][k] / scale);
          s += (M2[i][k] * M2[i][k]);
        }
        f = M2[i][l];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][l] = f - g;
        for (k = l; k < m; k++) rv1[k] = M2[i][k] / h;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = l; k < m; k++) s += (M2[j][k] * M2[i][k]);
            for (k = l; k < m; k++) M2[j][k] += (s * rv1[k]);
          }
        }
        for (k = l; k < m; k++) M2[i][k] = M2[i][k] * scale;
      }
    }
    anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
  }

  /* accumulate the right-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    if (i < m - 1) {
      if (g) {
        for (j = l; j < m; j++) v[j][i] = (M2[i][j] / M2[i][l]) / g;
        /* double division to avoid underflow */
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[i][k] * v[k][j]);
          for (k = l; k < m; k++) v[k][j] += (s * v[k][i]);
        }
      }
      for (j = l; j < m; j++) v[i][j] = v[j][i] = 0.0;
    }
    v[i][i] = 1.0;
    g = rv1[i];
    l = i;
  }

  /* accumulate the left-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    l = i + 1;
    g = w[i];
    if (i < m - 1)
      for (j = l; j < m; j++) M2[i][j] = 0.0;
    if (g) {
      g = 1.0 / g;
      if (i != m - 1) {
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[k][i] * M2[k][j]);
          f = (s / M2[i][i]) * g;
          for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
        }
      }
      for (j = i; j < m; j++) M2[j][i] = (M2[j][i] * g);
    } else {
      for (j = i; j < m; j++) M2[j][i] = 0.0;
    }
    ++M2[i][i];
  }

  /* diagonalize the bidiagonal form */
  for (k = m - 1; k >= 0; k--) {     /* loop over singular values */
    for (its = 0; its < 30; its++) { /* loop over allowed iterations */
      flag = 1;
      for (l = k; l >= 0; l--) { /* test for splitting */
        nm = l - 1;
        if (fabs(rv1[l]) + anorm == anorm) {
          flag = 0;
          break;
        }
        if (fabs(w[nm]) + anorm == anorm) break;
      }
      if (flag) {
        c = 0.0;
        s = 1.0;
        for (i = l; i <= k; i++) {
          f = s * rv1[i];
          if (fabs(f) + anorm != anorm) {
            g = w[i];
            h = PYTHAG(f, g);
            w[i] = h;
            h = 1.0 / h;
            c = g * h;
            s = (-f * h);
            for (j = 0; j < m; j++) {
              y = M2[j][nm];
              z = M2[j][i];
              M2[j][nm] = (y * c + z * s);
              M2[j][i] = (z * c - y * s);
            }
          }
        }
      }
      z = w[k];
      if (l == k) {    /* convergence */
        if (z < 0.0) { /* make singular value nonnegative */
          w[k] = (-z);
          for (j = 0; j < m; j++) v[j][k] = (-v[j][k]);
        }
        break;
      }
      if (its >= 30) {
        printf("Not converged\n");
      }

      /* shift from bottom 2 x 2 minor */
      x = w[l];
      nm = k - 1;
      y = w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = PYTHAG(f, 1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

      /* next QR transformation */
      c = s = 1.0;
      for (j = l; j <= nm; j++) {
        i = j + 1;
        g = rv1[i];
        y = w[i];
        h = s * g;
        g = c * g;
        z = PYTHAG(f, h);
        rv1[j] = z;
        c = f / z;
        s = h / z;
        f = x * c + g * s;
        g = g * c - x * s;
        h = y * s;
        y = y * c;
        for (jj = 0; jj < m; jj++) {
          x = v[jj][j];
          z = v[jj][i];
          v[jj][j] = (x * c + z * s);
          v[jj][i] = (z * c - x * s);
        }
        z = PYTHAG(f, h);
        w[j] = z;
        if (z) {
          z = 1.0 / z;
          c = f * z;
          s = h * z;
        }
        f = (c * g) + (s * y);
        x = (c * y) - (s * g);
        for (jj = 0; jj < m; jj++) {
          y = M2[jj][j];
          z = M2[jj][i];
          M2[jj][j] = (y * c + z * s);
          M2[jj][i] = (z * c - y * s);
        }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      w[k] = x;
    }
  }

  // start compute inverse matrix

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      res1[rowNumber][colNumber] = v[rowNumber][colNumber] / w[colNumber];
    }
  }

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      M2Inv[rowNumber][colNumber] = 0;
      for (int thirdIndex = 0; thirdIndex < m; thirdIndex++) {
        M2Inv[rowNumber][colNumber] =
            M2Inv[rowNumber][colNumber] + res1[rowNumber][thirdIndex] * M2[colNumber][thirdIndex];
      }
    }
  }
}

__device__ void pseudoinversel9(double M2[][9], double M2Inv[][9], double v[][9], double* rv1, double* w,
                                double res1[][9]) {
  int m = 9;
  int flag, its, i, j, jj, k, l, nm;
  double c, f, h, s, x, y, z;
  double anorm = 0.0, g = 0.0, scale = 0.0;
  /* Householder reduction to bidiagonal form */
  for (i = 0; i < m; i++) {
    /* left-hand reduction */
    l = i + 1;
    rv1[i] = scale * g;
    g = s = scale = 0.0;
    if (i < m) {
      for (k = i; k < m; k++) scale += fabs(M2[k][i]);
      if (scale) {
        for (k = i; k < m; k++) {
          M2[k][i] = (M2[k][i] / scale);
          s += (M2[k][i] * M2[k][i]);
        }
        f = M2[i][i];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][i] = f - g;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = i; k < m; k++) s += (M2[k][i] * M2[k][j]);
            f = s / h;
            for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
          }
        }
        for (k = i; k < m; k++) M2[k][i] = (M2[k][i] * scale);
      }
    }
    w[i] = scale * g;

    /* right-hand reduction */
    g = s = scale = 0.0;
    if (i < m && i != m - 1) {
      for (k = l; k < m; k++) scale += fabs(M2[i][k]);
      if (scale) {
        for (k = l; k < m; k++) {
          M2[i][k] = (M2[i][k] / scale);
          s += (M2[i][k] * M2[i][k]);
        }
        f = M2[i][l];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][l] = f - g;
        for (k = l; k < m; k++) rv1[k] = M2[i][k] / h;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = l; k < m; k++) s += (M2[j][k] * M2[i][k]);
            for (k = l; k < m; k++) M2[j][k] += (s * rv1[k]);
          }
        }
        for (k = l; k < m; k++) M2[i][k] = M2[i][k] * scale;
      }
    }
    anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
  }

  /* accumulate the right-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    if (i < m - 1) {
      if (g) {
        for (j = l; j < m; j++) v[j][i] = (M2[i][j] / M2[i][l]) / g;
        /* double division to avoid underflow */
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[i][k] * v[k][j]);
          for (k = l; k < m; k++) v[k][j] += (s * v[k][i]);
        }
      }
      for (j = l; j < m; j++) v[i][j] = v[j][i] = 0.0;
    }
    v[i][i] = 1.0;
    g = rv1[i];
    l = i;
  }

  /* accumulate the left-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    l = i + 1;
    g = w[i];
    if (i < m - 1)
      for (j = l; j < m; j++) M2[i][j] = 0.0;
    if (g) {
      g = 1.0 / g;
      if (i != m - 1) {
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[k][i] * M2[k][j]);
          f = (s / M2[i][i]) * g;
          for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
        }
      }
      for (j = i; j < m; j++) M2[j][i] = (M2[j][i] * g);
    } else {
      for (j = i; j < m; j++) M2[j][i] = 0.0;
    }
    ++M2[i][i];
  }

  /* diagonalize the bidiagonal form */
  for (k = m - 1; k >= 0; k--) {     /* loop over singular values */
    for (its = 0; its < 30; its++) { /* loop over allowed iterations */
      flag = 1;
      for (l = k; l >= 0; l--) { /* test for splitting */
        nm = l - 1;
        if (fabs(rv1[l]) + anorm == anorm) {
          flag = 0;
          break;
        }
        if (fabs(w[nm]) + anorm == anorm) break;
      }
      if (flag) {
        c = 0.0;
        s = 1.0;
        for (i = l; i <= k; i++) {
          f = s * rv1[i];
          if (fabs(f) + anorm != anorm) {
            g = w[i];
            h = PYTHAG(f, g);
            w[i] = h;
            h = 1.0 / h;
            c = g * h;
            s = (-f * h);
            for (j = 0; j < m; j++) {
              y = M2[j][nm];
              z = M2[j][i];
              M2[j][nm] = (y * c + z * s);
              M2[j][i] = (z * c - y * s);
            }
          }
        }
      }
      z = w[k];
      if (l == k) {    /* convergence */
        if (z < 0.0) { /* make singular value nonnegative */
          w[k] = (-z);
          for (j = 0; j < m; j++) v[j][k] = (-v[j][k]);
        }
        break;
      }
      if (its >= 30) {
        printf("Not converged\n");
      }

      /* shift from bottom 2 x 2 minor */
      x = w[l];
      nm = k - 1;
      y = w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = PYTHAG(f, 1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

      /* next QR transformation */
      c = s = 1.0;
      for (j = l; j <= nm; j++) {
        i = j + 1;
        g = rv1[i];
        y = w[i];
        h = s * g;
        g = c * g;
        z = PYTHAG(f, h);
        rv1[j] = z;
        c = f / z;
        s = h / z;
        f = x * c + g * s;
        g = g * c - x * s;
        h = y * s;
        y = y * c;
        for (jj = 0; jj < m; jj++) {
          x = v[jj][j];
          z = v[jj][i];
          v[jj][j] = (x * c + z * s);
          v[jj][i] = (z * c - x * s);
        }
        z = PYTHAG(f, h);
        w[j] = z;
        if (z) {
          z = 1.0 / z;
          c = f * z;
          s = h * z;
        }
        f = (c * g) + (s * y);
        x = (c * y) - (s * g);
        for (jj = 0; jj < m; jj++) {
          y = M2[jj][j];
          z = M2[jj][i];
          M2[jj][j] = (y * c + z * s);
          M2[jj][i] = (z * c - y * s);
        }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      w[k] = x;
    }
  }

  // start compute inverse matrix

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      res1[rowNumber][colNumber] = v[rowNumber][colNumber] / w[colNumber];
    }
  }

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      M2Inv[rowNumber][colNumber] = 0;
      for (int thirdIndex = 0; thirdIndex < m; thirdIndex++) {
        M2Inv[rowNumber][colNumber] =
            M2Inv[rowNumber][colNumber] + res1[rowNumber][thirdIndex] * M2[colNumber][thirdIndex];
      }
    }
  }
}

__device__ void pseudoinversel10(double M2[][10], double M2Inv[][10], double v[][10], double* rv1, double* w,
                                 double res1[][10]) {
  int m = 10;
  int flag, its, i, j, jj, k, l, nm;
  double c, f, h, s, x, y, z;
  double anorm = 0.0, g = 0.0, scale = 0.0;
  /* Householder reduction to bidiagonal form */
  for (i = 0; i < m; i++) {
    /* left-hand reduction */
    l = i + 1;
    rv1[i] = scale * g;
    g = s = scale = 0.0;
    if (i < m) {
      for (k = i; k < m; k++) scale += fabs(M2[k][i]);
      if (scale) {
        for (k = i; k < m; k++) {
          M2[k][i] = (M2[k][i] / scale);
          s += (M2[k][i] * M2[k][i]);
        }
        f = M2[i][i];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][i] = f - g;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = i; k < m; k++) s += (M2[k][i] * M2[k][j]);
            f = s / h;
            for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
          }
        }
        for (k = i; k < m; k++) M2[k][i] = (M2[k][i] * scale);
      }
    }
    w[i] = scale * g;

    /* right-hand reduction */
    g = s = scale = 0.0;
    if (i < m && i != m - 1) {
      for (k = l; k < m; k++) scale += fabs(M2[i][k]);
      if (scale) {
        for (k = l; k < m; k++) {
          M2[i][k] = (M2[i][k] / scale);
          s += (M2[i][k] * M2[i][k]);
        }
        f = M2[i][l];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][l] = f - g;
        for (k = l; k < m; k++) rv1[k] = M2[i][k] / h;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = l; k < m; k++) s += (M2[j][k] * M2[i][k]);
            for (k = l; k < m; k++) M2[j][k] += (s * rv1[k]);
          }
        }
        for (k = l; k < m; k++) M2[i][k] = M2[i][k] * scale;
      }
    }
    anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
  }

  /* accumulate the right-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    if (i < m - 1) {
      if (g) {
        for (j = l; j < m; j++) v[j][i] = (M2[i][j] / M2[i][l]) / g;
        /* double division to avoid underflow */
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[i][k] * v[k][j]);
          for (k = l; k < m; k++) v[k][j] += (s * v[k][i]);
        }
      }
      for (j = l; j < m; j++) v[i][j] = v[j][i] = 0.0;
    }
    v[i][i] = 1.0;
    g = rv1[i];
    l = i;
  }

  /* accumulate the left-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    l = i + 1;
    g = w[i];
    if (i < m - 1)
      for (j = l; j < m; j++) M2[i][j] = 0.0;
    if (g) {
      g = 1.0 / g;
      if (i != m - 1) {
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[k][i] * M2[k][j]);
          f = (s / M2[i][i]) * g;
          for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
        }
      }
      for (j = i; j < m; j++) M2[j][i] = (M2[j][i] * g);
    } else {
      for (j = i; j < m; j++) M2[j][i] = 0.0;
    }
    ++M2[i][i];
  }

  /* diagonalize the bidiagonal form */
  for (k = m - 1; k >= 0; k--) {     /* loop over singular values */
    for (its = 0; its < 30; its++) { /* loop over allowed iterations */
      flag = 1;
      for (l = k; l >= 0; l--) { /* test for splitting */
        nm = l - 1;
        if (fabs(rv1[l]) + anorm == anorm) {
          flag = 0;
          break;
        }
        if (fabs(w[nm]) + anorm == anorm) break;
      }
      if (flag) {
        c = 0.0;
        s = 1.0;
        for (i = l; i <= k; i++) {
          f = s * rv1[i];
          if (fabs(f) + anorm != anorm) {
            g = w[i];
            h = PYTHAG(f, g);
            w[i] = h;
            h = 1.0 / h;
            c = g * h;
            s = (-f * h);
            for (j = 0; j < m; j++) {
              y = M2[j][nm];
              z = M2[j][i];
              M2[j][nm] = (y * c + z * s);
              M2[j][i] = (z * c - y * s);
            }
          }
        }
      }
      z = w[k];
      if (l == k) {    /* convergence */
        if (z < 0.0) { /* make singular value nonnegative */
          w[k] = (-z);
          for (j = 0; j < m; j++) v[j][k] = (-v[j][k]);
        }
        break;
      }
      if (its >= 30) {
        printf("Not converged\n");
      }

      /* shift from bottom 2 x 2 minor */
      x = w[l];
      nm = k - 1;
      y = w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = PYTHAG(f, 1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

      /* next QR transformation */
      c = s = 1.0;
      for (j = l; j <= nm; j++) {
        i = j + 1;
        g = rv1[i];
        y = w[i];
        h = s * g;
        g = c * g;
        z = PYTHAG(f, h);
        rv1[j] = z;
        c = f / z;
        s = h / z;
        f = x * c + g * s;
        g = g * c - x * s;
        h = y * s;
        y = y * c;
        for (jj = 0; jj < m; jj++) {
          x = v[jj][j];
          z = v[jj][i];
          v[jj][j] = (x * c + z * s);
          v[jj][i] = (z * c - x * s);
        }
        z = PYTHAG(f, h);
        w[j] = z;
        if (z) {
          z = 1.0 / z;
          c = f * z;
          s = h * z;
        }
        f = (c * g) + (s * y);
        x = (c * y) - (s * g);
        for (jj = 0; jj < m; jj++) {
          y = M2[jj][j];
          z = M2[jj][i];
          M2[jj][j] = (y * c + z * s);
          M2[jj][i] = (z * c - y * s);
        }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      w[k] = x;
    }
  }

  // start compute inverse matrix

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      res1[rowNumber][colNumber] = v[rowNumber][colNumber] / w[colNumber];
    }
  }

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      M2Inv[rowNumber][colNumber] = 0;
      for (int thirdIndex = 0; thirdIndex < m; thirdIndex++) {
        M2Inv[rowNumber][colNumber] =
            M2Inv[rowNumber][colNumber] + res1[rowNumber][thirdIndex] * M2[colNumber][thirdIndex];
      }
    }
  }
}

__device__ void pseudoinversel11(double M2[][11], double M2Inv[][11], double v[][11], double* rv1, double* w,
                                 double res1[][11]) {
  int m = 11;
  int flag, its, i, j, jj, k, l, nm;
  double c, f, h, s, x, y, z;
  double anorm = 0.0, g = 0.0, scale = 0.0;
  /* Householder reduction to bidiagonal form */
  for (i = 0; i < m; i++) {
    /* left-hand reduction */
    l = i + 1;
    rv1[i] = scale * g;
    g = s = scale = 0.0;
    if (i < m) {
      for (k = i; k < m; k++) scale += fabs(M2[k][i]);
      if (scale) {
        for (k = i; k < m; k++) {
          M2[k][i] = (M2[k][i] / scale);
          s += (M2[k][i] * M2[k][i]);
        }
        f = M2[i][i];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][i] = f - g;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = i; k < m; k++) s += (M2[k][i] * M2[k][j]);
            f = s / h;
            for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
          }
        }
        for (k = i; k < m; k++) M2[k][i] = (M2[k][i] * scale);
      }
    }
    w[i] = scale * g;

    /* right-hand reduction */
    g = s = scale = 0.0;
    if (i < m && i != m - 1) {
      for (k = l; k < m; k++) scale += fabs(M2[i][k]);
      if (scale) {
        for (k = l; k < m; k++) {
          M2[i][k] = (M2[i][k] / scale);
          s += (M2[i][k] * M2[i][k]);
        }
        f = M2[i][l];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][l] = f - g;
        for (k = l; k < m; k++) rv1[k] = M2[i][k] / h;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = l; k < m; k++) s += (M2[j][k] * M2[i][k]);
            for (k = l; k < m; k++) M2[j][k] += (s * rv1[k]);
          }
        }
        for (k = l; k < m; k++) M2[i][k] = M2[i][k] * scale;
      }
    }
    anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
  }

  /* accumulate the right-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    if (i < m - 1) {
      if (g) {
        for (j = l; j < m; j++) v[j][i] = (M2[i][j] / M2[i][l]) / g;
        /* double division to avoid underflow */
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[i][k] * v[k][j]);
          for (k = l; k < m; k++) v[k][j] += (s * v[k][i]);
        }
      }
      for (j = l; j < m; j++) v[i][j] = v[j][i] = 0.0;
    }
    v[i][i] = 1.0;
    g = rv1[i];
    l = i;
  }

  /* accumulate the left-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    l = i + 1;
    g = w[i];
    if (i < m - 1)
      for (j = l; j < m; j++) M2[i][j] = 0.0;
    if (g) {
      g = 1.0 / g;
      if (i != m - 1) {
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[k][i] * M2[k][j]);
          f = (s / M2[i][i]) * g;
          for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
        }
      }
      for (j = i; j < m; j++) M2[j][i] = (M2[j][i] * g);
    } else {
      for (j = i; j < m; j++) M2[j][i] = 0.0;
    }
    ++M2[i][i];
  }

  /* diagonalize the bidiagonal form */
  for (k = m - 1; k >= 0; k--) {     /* loop over singular values */
    for (its = 0; its < 30; its++) { /* loop over allowed iterations */
      flag = 1;
      for (l = k; l >= 0; l--) { /* test for splitting */
        nm = l - 1;
        if (fabs(rv1[l]) + anorm == anorm) {
          flag = 0;
          break;
        }
        if (fabs(w[nm]) + anorm == anorm) break;
      }
      if (flag) {
        c = 0.0;
        s = 1.0;
        for (i = l; i <= k; i++) {
          f = s * rv1[i];
          if (fabs(f) + anorm != anorm) {
            g = w[i];
            h = PYTHAG(f, g);
            w[i] = h;
            h = 1.0 / h;
            c = g * h;
            s = (-f * h);
            for (j = 0; j < m; j++) {
              y = M2[j][nm];
              z = M2[j][i];
              M2[j][nm] = (y * c + z * s);
              M2[j][i] = (z * c - y * s);
            }
          }
        }
      }
      z = w[k];
      if (l == k) {    /* convergence */
        if (z < 0.0) { /* make singular value nonnegative */
          w[k] = (-z);
          for (j = 0; j < m; j++) v[j][k] = (-v[j][k]);
        }
        break;
      }
      if (its >= 30) {
        printf("Not converged\n");
      }

      /* shift from bottom 2 x 2 minor */
      x = w[l];
      nm = k - 1;
      y = w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = PYTHAG(f, 1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

      /* next QR transformation */
      c = s = 1.0;
      for (j = l; j <= nm; j++) {
        i = j + 1;
        g = rv1[i];
        y = w[i];
        h = s * g;
        g = c * g;
        z = PYTHAG(f, h);
        rv1[j] = z;
        c = f / z;
        s = h / z;
        f = x * c + g * s;
        g = g * c - x * s;
        h = y * s;
        y = y * c;
        for (jj = 0; jj < m; jj++) {
          x = v[jj][j];
          z = v[jj][i];
          v[jj][j] = (x * c + z * s);
          v[jj][i] = (z * c - x * s);
        }
        z = PYTHAG(f, h);
        w[j] = z;
        if (z) {
          z = 1.0 / z;
          c = f * z;
          s = h * z;
        }
        f = (c * g) + (s * y);
        x = (c * y) - (s * g);
        for (jj = 0; jj < m; jj++) {
          y = M2[jj][j];
          z = M2[jj][i];
          M2[jj][j] = (y * c + z * s);
          M2[jj][i] = (z * c - y * s);
        }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      w[k] = x;
    }
  }

  // start compute inverse matrix

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      res1[rowNumber][colNumber] = v[rowNumber][colNumber] / w[colNumber];
    }
  }

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      M2Inv[rowNumber][colNumber] = 0;
      for (int thirdIndex = 0; thirdIndex < m; thirdIndex++) {
        M2Inv[rowNumber][colNumber] =
            M2Inv[rowNumber][colNumber] + res1[rowNumber][thirdIndex] * M2[colNumber][thirdIndex];
      }
    }
  }
}

__device__ void pseudoinversel12(double M2[][12], double M2Inv[][12], double v[][12], double* rv1, double* w,
                                 double res1[][12]) {
  int m = 12;
  int flag, its, i, j, jj, k, l, nm;
  double c, f, h, s, x, y, z;
  double anorm = 0.0, g = 0.0, scale = 0.0;
  /* Householder reduction to bidiagonal form */
  for (i = 0; i < m; i++) {
    /* left-hand reduction */
    l = i + 1;
    rv1[i] = scale * g;
    g = s = scale = 0.0;
    if (i < m) {
      for (k = i; k < m; k++) scale += fabs(M2[k][i]);
      if (scale) {
        for (k = i; k < m; k++) {
          M2[k][i] = (M2[k][i] / scale);
          s += (M2[k][i] * M2[k][i]);
        }
        f = M2[i][i];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][i] = f - g;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = i; k < m; k++) s += (M2[k][i] * M2[k][j]);
            f = s / h;
            for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
          }
        }
        for (k = i; k < m; k++) M2[k][i] = (M2[k][i] * scale);
      }
    }
    w[i] = scale * g;

    /* right-hand reduction */
    g = s = scale = 0.0;
    if (i < m && i != m - 1) {
      for (k = l; k < m; k++) scale += fabs(M2[i][k]);
      if (scale) {
        for (k = l; k < m; k++) {
          M2[i][k] = (M2[i][k] / scale);
          s += (M2[i][k] * M2[i][k]);
        }
        f = M2[i][l];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][l] = f - g;
        for (k = l; k < m; k++) rv1[k] = M2[i][k] / h;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = l; k < m; k++) s += (M2[j][k] * M2[i][k]);
            for (k = l; k < m; k++) M2[j][k] += (s * rv1[k]);
          }
        }
        for (k = l; k < m; k++) M2[i][k] = M2[i][k] * scale;
      }
    }
    anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
  }

  /* accumulate the right-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    if (i < m - 1) {
      if (g) {
        for (j = l; j < m; j++) v[j][i] = (M2[i][j] / M2[i][l]) / g;
        /* double division to avoid underflow */
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[i][k] * v[k][j]);
          for (k = l; k < m; k++) v[k][j] += (s * v[k][i]);
        }
      }
      for (j = l; j < m; j++) v[i][j] = v[j][i] = 0.0;
    }
    v[i][i] = 1.0;
    g = rv1[i];
    l = i;
  }

  /* accumulate the left-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    l = i + 1;
    g = w[i];
    if (i < m - 1)
      for (j = l; j < m; j++) M2[i][j] = 0.0;
    if (g) {
      g = 1.0 / g;
      if (i != m - 1) {
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[k][i] * M2[k][j]);
          f = (s / M2[i][i]) * g;
          for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
        }
      }
      for (j = i; j < m; j++) M2[j][i] = (M2[j][i] * g);
    } else {
      for (j = i; j < m; j++) M2[j][i] = 0.0;
    }
    ++M2[i][i];
  }

  /* diagonalize the bidiagonal form */
  for (k = m - 1; k >= 0; k--) {     /* loop over singular values */
    for (its = 0; its < 30; its++) { /* loop over allowed iterations */
      flag = 1;
      for (l = k; l >= 0; l--) { /* test for splitting */
        nm = l - 1;
        if (fabs(rv1[l]) + anorm == anorm) {
          flag = 0;
          break;
        }
        if (fabs(w[nm]) + anorm == anorm) break;
      }
      if (flag) {
        c = 0.0;
        s = 1.0;
        for (i = l; i <= k; i++) {
          f = s * rv1[i];
          if (fabs(f) + anorm != anorm) {
            g = w[i];
            h = PYTHAG(f, g);
            w[i] = h;
            h = 1.0 / h;
            c = g * h;
            s = (-f * h);
            for (j = 0; j < m; j++) {
              y = M2[j][nm];
              z = M2[j][i];
              M2[j][nm] = (y * c + z * s);
              M2[j][i] = (z * c - y * s);
            }
          }
        }
      }
      z = w[k];
      if (l == k) {    /* convergence */
        if (z < 0.0) { /* make singular value nonnegative */
          w[k] = (-z);
          for (j = 0; j < m; j++) v[j][k] = (-v[j][k]);
        }
        break;
      }
      if (its >= 30) {
        printf("Not converged\n");
      }

      /* shift from bottom 2 x 2 minor */
      x = w[l];
      nm = k - 1;
      y = w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = PYTHAG(f, 1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

      /* next QR transformation */
      c = s = 1.0;
      for (j = l; j <= nm; j++) {
        i = j + 1;
        g = rv1[i];
        y = w[i];
        h = s * g;
        g = c * g;
        z = PYTHAG(f, h);
        rv1[j] = z;
        c = f / z;
        s = h / z;
        f = x * c + g * s;
        g = g * c - x * s;
        h = y * s;
        y = y * c;
        for (jj = 0; jj < m; jj++) {
          x = v[jj][j];
          z = v[jj][i];
          v[jj][j] = (x * c + z * s);
          v[jj][i] = (z * c - x * s);
        }
        z = PYTHAG(f, h);
        w[j] = z;
        if (z) {
          z = 1.0 / z;
          c = f * z;
          s = h * z;
        }
        f = (c * g) + (s * y);
        x = (c * y) - (s * g);
        for (jj = 0; jj < m; jj++) {
          y = M2[jj][j];
          z = M2[jj][i];
          M2[jj][j] = (y * c + z * s);
          M2[jj][i] = (z * c - y * s);
        }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      w[k] = x;
    }
  }

  // start compute inverse matrix

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      res1[rowNumber][colNumber] = v[rowNumber][colNumber] / w[colNumber];
    }
  }

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      M2Inv[rowNumber][colNumber] = 0;
      for (int thirdIndex = 0; thirdIndex < m; thirdIndex++) {
        M2Inv[rowNumber][colNumber] =
            M2Inv[rowNumber][colNumber] + res1[rowNumber][thirdIndex] * M2[colNumber][thirdIndex];
      }
    }
  }
}

__device__ void pseudoinversel13(double M2[][13], double M2Inv[][13], double v[][13], double* rv1, double* w,
                                 double res1[][13]) {
  int m = 13;
  int flag, its, i, j, jj, k, l, nm;
  double c, f, h, s, x, y, z;
  double anorm = 0.0, g = 0.0, scale = 0.0;
  /* Householder reduction to bidiagonal form */
  for (i = 0; i < m; i++) {
    /* left-hand reduction */
    l = i + 1;
    rv1[i] = scale * g;
    g = s = scale = 0.0;
    if (i < m) {
      for (k = i; k < m; k++) scale += fabs(M2[k][i]);
      if (scale) {
        for (k = i; k < m; k++) {
          M2[k][i] = (M2[k][i] / scale);
          s += (M2[k][i] * M2[k][i]);
        }
        f = M2[i][i];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][i] = f - g;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = i; k < m; k++) s += (M2[k][i] * M2[k][j]);
            f = s / h;
            for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
          }
        }
        for (k = i; k < m; k++) M2[k][i] = (M2[k][i] * scale);
      }
    }
    w[i] = scale * g;

    /* right-hand reduction */
    g = s = scale = 0.0;
    if (i < m && i != m - 1) {
      for (k = l; k < m; k++) scale += fabs(M2[i][k]);
      if (scale) {
        for (k = l; k < m; k++) {
          M2[i][k] = (M2[i][k] / scale);
          s += (M2[i][k] * M2[i][k]);
        }
        f = M2[i][l];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][l] = f - g;
        for (k = l; k < m; k++) rv1[k] = M2[i][k] / h;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = l; k < m; k++) s += (M2[j][k] * M2[i][k]);
            for (k = l; k < m; k++) M2[j][k] += (s * rv1[k]);
          }
        }
        for (k = l; k < m; k++) M2[i][k] = M2[i][k] * scale;
      }
    }
    anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
  }

  /* accumulate the right-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    if (i < m - 1) {
      if (g) {
        for (j = l; j < m; j++) v[j][i] = (M2[i][j] / M2[i][l]) / g;
        /* double division to avoid underflow */
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[i][k] * v[k][j]);
          for (k = l; k < m; k++) v[k][j] += (s * v[k][i]);
        }
      }
      for (j = l; j < m; j++) v[i][j] = v[j][i] = 0.0;
    }
    v[i][i] = 1.0;
    g = rv1[i];
    l = i;
  }

  /* accumulate the left-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    l = i + 1;
    g = w[i];
    if (i < m - 1)
      for (j = l; j < m; j++) M2[i][j] = 0.0;
    if (g) {
      g = 1.0 / g;
      if (i != m - 1) {
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[k][i] * M2[k][j]);
          f = (s / M2[i][i]) * g;
          for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
        }
      }
      for (j = i; j < m; j++) M2[j][i] = (M2[j][i] * g);
    } else {
      for (j = i; j < m; j++) M2[j][i] = 0.0;
    }
    ++M2[i][i];
  }

  /* diagonalize the bidiagonal form */
  for (k = m - 1; k >= 0; k--) {     /* loop over singular values */
    for (its = 0; its < 30; its++) { /* loop over allowed iterations */
      flag = 1;
      for (l = k; l >= 0; l--) { /* test for splitting */
        nm = l - 1;
        if (fabs(rv1[l]) + anorm == anorm) {
          flag = 0;
          break;
        }
        if (fabs(w[nm]) + anorm == anorm) break;
      }
      if (flag) {
        c = 0.0;
        s = 1.0;
        for (i = l; i <= k; i++) {
          f = s * rv1[i];
          if (fabs(f) + anorm != anorm) {
            g = w[i];
            h = PYTHAG(f, g);
            w[i] = h;
            h = 1.0 / h;
            c = g * h;
            s = (-f * h);
            for (j = 0; j < m; j++) {
              y = M2[j][nm];
              z = M2[j][i];
              M2[j][nm] = (y * c + z * s);
              M2[j][i] = (z * c - y * s);
            }
          }
        }
      }
      z = w[k];
      if (l == k) {    /* convergence */
        if (z < 0.0) { /* make singular value nonnegative */
          w[k] = (-z);
          for (j = 0; j < m; j++) v[j][k] = (-v[j][k]);
        }
        break;
      }
      if (its >= 30) {
        printf("Not converged\n");
      }

      /* shift from bottom 2 x 2 minor */
      x = w[l];
      nm = k - 1;
      y = w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = PYTHAG(f, 1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

      /* next QR transformation */
      c = s = 1.0;
      for (j = l; j <= nm; j++) {
        i = j + 1;
        g = rv1[i];
        y = w[i];
        h = s * g;
        g = c * g;
        z = PYTHAG(f, h);
        rv1[j] = z;
        c = f / z;
        s = h / z;
        f = x * c + g * s;
        g = g * c - x * s;
        h = y * s;
        y = y * c;
        for (jj = 0; jj < m; jj++) {
          x = v[jj][j];
          z = v[jj][i];
          v[jj][j] = (x * c + z * s);
          v[jj][i] = (z * c - x * s);
        }
        z = PYTHAG(f, h);
        w[j] = z;
        if (z) {
          z = 1.0 / z;
          c = f * z;
          s = h * z;
        }
        f = (c * g) + (s * y);
        x = (c * y) - (s * g);
        for (jj = 0; jj < m; jj++) {
          y = M2[jj][j];
          z = M2[jj][i];
          M2[jj][j] = (y * c + z * s);
          M2[jj][i] = (z * c - y * s);
        }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      w[k] = x;
    }
  }

  // start compute inverse matrix

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      res1[rowNumber][colNumber] = v[rowNumber][colNumber] / w[colNumber];
    }
  }

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      M2Inv[rowNumber][colNumber] = 0;
      for (int thirdIndex = 0; thirdIndex < m; thirdIndex++) {
        M2Inv[rowNumber][colNumber] =
            M2Inv[rowNumber][colNumber] + res1[rowNumber][thirdIndex] * M2[colNumber][thirdIndex];
      }
    }
  }
}

__device__ void pseudoinversel14(double M2[][14], double M2Inv[][14], double v[][14], double* rv1, double* w,
                                 double res1[][14]) {
  int m = 14;
  int flag, its, i, j, jj, k, l, nm;
  double c, f, h, s, x, y, z;
  double anorm = 0.0, g = 0.0, scale = 0.0;
  /* Householder reduction to bidiagonal form */
  for (i = 0; i < m; i++) {
    /* left-hand reduction */
    l = i + 1;
    rv1[i] = scale * g;
    g = s = scale = 0.0;
    if (i < m) {
      for (k = i; k < m; k++) scale += fabs(M2[k][i]);
      if (scale) {
        for (k = i; k < m; k++) {
          M2[k][i] = (M2[k][i] / scale);
          s += (M2[k][i] * M2[k][i]);
        }
        f = M2[i][i];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][i] = f - g;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = i; k < m; k++) s += (M2[k][i] * M2[k][j]);
            f = s / h;
            for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
          }
        }
        for (k = i; k < m; k++) M2[k][i] = (M2[k][i] * scale);
      }
    }
    w[i] = scale * g;

    /* right-hand reduction */
    g = s = scale = 0.0;
    if (i < m && i != m - 1) {
      for (k = l; k < m; k++) scale += fabs(M2[i][k]);
      if (scale) {
        for (k = l; k < m; k++) {
          M2[i][k] = (M2[i][k] / scale);
          s += (M2[i][k] * M2[i][k]);
        }
        f = M2[i][l];
        g = -SIGN(sqrt(s), f);
        h = f * g - s;
        M2[i][l] = f - g;
        for (k = l; k < m; k++) rv1[k] = M2[i][k] / h;
        if (i != m - 1) {
          for (j = l; j < m; j++) {
            for (s = 0.0, k = l; k < m; k++) s += (M2[j][k] * M2[i][k]);
            for (k = l; k < m; k++) M2[j][k] += (s * rv1[k]);
          }
        }
        for (k = l; k < m; k++) M2[i][k] = M2[i][k] * scale;
      }
    }
    anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
  }

  /* accumulate the right-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    if (i < m - 1) {
      if (g) {
        for (j = l; j < m; j++) v[j][i] = (M2[i][j] / M2[i][l]) / g;
        /* double division to avoid underflow */
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[i][k] * v[k][j]);
          for (k = l; k < m; k++) v[k][j] += (s * v[k][i]);
        }
      }
      for (j = l; j < m; j++) v[i][j] = v[j][i] = 0.0;
    }
    v[i][i] = 1.0;
    g = rv1[i];
    l = i;
  }

  /* accumulate the left-hand transformation */
  for (i = m - 1; i >= 0; i--) {
    l = i + 1;
    g = w[i];
    if (i < m - 1)
      for (j = l; j < m; j++) M2[i][j] = 0.0;
    if (g) {
      g = 1.0 / g;
      if (i != m - 1) {
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < m; k++) s += (M2[k][i] * M2[k][j]);
          f = (s / M2[i][i]) * g;
          for (k = i; k < m; k++) M2[k][j] += (f * M2[k][i]);
        }
      }
      for (j = i; j < m; j++) M2[j][i] = (M2[j][i] * g);
    } else {
      for (j = i; j < m; j++) M2[j][i] = 0.0;
    }
    ++M2[i][i];
  }

  /* diagonalize the bidiagonal form */
  for (k = m - 1; k >= 0; k--) {     /* loop over singular values */
    for (its = 0; its < 30; its++) { /* loop over allowed iterations */
      flag = 1;
      for (l = k; l >= 0; l--) { /* test for splitting */
        nm = l - 1;
        if (fabs(rv1[l]) + anorm == anorm) {
          flag = 0;
          break;
        }
        if (fabs(w[nm]) + anorm == anorm) break;
      }
      if (flag) {
        c = 0.0;
        s = 1.0;
        for (i = l; i <= k; i++) {
          f = s * rv1[i];
          if (fabs(f) + anorm != anorm) {
            g = w[i];
            h = PYTHAG(f, g);
            w[i] = h;
            h = 1.0 / h;
            c = g * h;
            s = (-f * h);
            for (j = 0; j < m; j++) {
              y = M2[j][nm];
              z = M2[j][i];
              M2[j][nm] = (y * c + z * s);
              M2[j][i] = (z * c - y * s);
            }
          }
        }
      }
      z = w[k];
      if (l == k) {    /* convergence */
        if (z < 0.0) { /* make singular value nonnegative */
          w[k] = (-z);
          for (j = 0; j < m; j++) v[j][k] = (-v[j][k]);
        }
        break;
      }
      if (its >= 30) {
        printf("Not converged\n");
      }

      /* shift from bottom 2 x 2 minor */
      x = w[l];
      nm = k - 1;
      y = w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = PYTHAG(f, 1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;

      /* next QR transformation */
      c = s = 1.0;
      for (j = l; j <= nm; j++) {
        i = j + 1;
        g = rv1[i];
        y = w[i];
        h = s * g;
        g = c * g;
        z = PYTHAG(f, h);
        rv1[j] = z;
        c = f / z;
        s = h / z;
        f = x * c + g * s;
        g = g * c - x * s;
        h = y * s;
        y = y * c;
        for (jj = 0; jj < m; jj++) {
          x = v[jj][j];
          z = v[jj][i];
          v[jj][j] = (x * c + z * s);
          v[jj][i] = (z * c - x * s);
        }
        z = PYTHAG(f, h);
        w[j] = z;
        if (z) {
          z = 1.0 / z;
          c = f * z;
          s = h * z;
        }
        f = (c * g) + (s * y);
        x = (c * y) - (s * g);
        for (jj = 0; jj < m; jj++) {
          y = M2[jj][j];
          z = M2[jj][i];
          M2[jj][j] = (y * c + z * s);
          M2[jj][i] = (z * c - y * s);
        }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      w[k] = x;
    }
  }

  // start compute inverse matrix

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      res1[rowNumber][colNumber] = v[rowNumber][colNumber] / w[colNumber];
    }
  }

  for (int rowNumber = 0; rowNumber < m; rowNumber++) {
    for (int colNumber = 0; colNumber < m; colNumber++) {
      M2Inv[rowNumber][colNumber] = 0;
      for (int thirdIndex = 0; thirdIndex < m; thirdIndex++) {
        M2Inv[rowNumber][colNumber] =
            M2Inv[rowNumber][colNumber] + res1[rowNumber][thirdIndex] * M2[colNumber][thirdIndex];
      }
    }
  }
}

__global__ void scan_compact(int* G_Compact, const int* G, const int n, int* nprime) {
  const int row = blockIdx.y;
  const int section = (n + blockDim.x - 1) / blockDim.x;
  int thid = 0;
  int tmp = 0;
  int stepSize = 0;
  extern __shared__ int G_shared[];
  // copy a row of data into shared memory
  for (int cnt = 0; cnt < section; cnt++) {
    thid = threadIdx.x + blockDim.x * cnt;
    if (thid < n) {
      G_shared[thid] = G[row * n + thid];
    }
  }

  __syncthreads();
  for (int sec = 0; sec < section; sec++) {
    thid = threadIdx.x + blockDim.x * sec;
    stepSize = ((n - sec * blockDim.x) / blockDim.x) > 0 ? blockDim.x : (n - sec * blockDim.x);
    for (int step = 1; step < stepSize; step = step * 2) {
      if (thid < n) {
        if (threadIdx.x < step) {
          tmp = G_shared[thid];
        } else if (threadIdx.x >= step) {
          tmp = G_shared[thid] + G_shared[thid - step];
        }
      }
      __syncthreads();
      if (thid < n) {
        G_shared[thid] = tmp;
      }
      __syncthreads();
    }
    if (thid == (blockDim.x * (sec + 1) - 1) && sec != (section - 1)) {
      G_shared[thid + 1] = G_shared[thid + 1] + G_shared[thid];
    }
    __syncthreads();
  }
  // ===============> Compact <===============
  const int row_size = G_shared[n - 1];

  for (int sec = 0; sec < section; sec++) {
    thid = threadIdx.x + blockDim.x * sec;
    if (thid < n && thid > 0) {
      if (G_shared[thid] != G_shared[thid - 1]) {
        G_Compact[row * n + G_shared[thid] - 1] = thid;
      }
      if (thid >= row_size && thid != n - 1) {
        G_Compact[row * n + thid] = 0;
      }
      if (thid == n - 1) {
        atomicMax(nprime, G_shared[n - 1]);
        G_Compact[row * n + n - 1] = G_shared[n - 1];
      }
    }
  }

  if (threadIdx.x == 0 && G[row * n] == 1) {
    G_Compact[row * n] = 0;
  }
}

__device__ void inverse(double M2[][3], double M2Inv[][3]) {
  double det = M2[0][0] * (M2[2][2] * M2[1][1]) - M2[0][0] * (M2[2][1] * M2[1][2]) - M2[1][0] * (M2[2][2] * M2[0][1]) +
               M2[1][0] * (M2[2][1] * M2[0][2]) + M2[2][0] * (M2[1][2] * M2[0][1]) - M2[2][0] * (M2[1][1] * M2[0][2]);
  double tmp = 1.0 / det;
  M2Inv[0][0] = tmp * (M2[1][1] * M2[2][2] - M2[1][2] * M2[2][1]);
  M2Inv[0][1] = tmp * (M2[0][2] * M2[2][1] - M2[0][1] * M2[2][2]);
  M2Inv[0][2] = tmp * (M2[0][1] * M2[1][2] - M2[0][2] * M2[1][1]);

  M2Inv[1][0] = tmp * (M2[1][2] * M2[2][0] - M2[1][0] * M2[2][2]);
  M2Inv[1][1] = tmp * (M2[0][0] * M2[2][2] - M2[0][2] * M2[2][0]);
  M2Inv[1][2] = tmp * (M2[0][2] * M2[1][0] - M2[0][0] * M2[1][2]);

  M2Inv[2][0] = tmp * (M2[1][0] * M2[2][1] - M2[1][1] * M2[2][0]);
  M2Inv[2][1] = tmp * (M2[0][1] * M2[2][0] - M2[0][0] * M2[2][1]);
  M2Inv[2][2] = tmp * (M2[0][0] * M2[1][1] - M2[0][1] * M2[1][0]);
}

__device__ void BINOM(int n, int k, int* out) {
  int P, N1, R;
  // between n - k and k, N1 should be Max(n-k, k) and P should be Min(n-k, k);
  N1 = k;
  P = n - k;
  if (N1 <= P) {
    N1 = P;
    P = k;
  }
  if (P == 0) {
    R = 1;
  } else if (P == 1) {
    R = N1 + 1;
  } else {
    R = N1 + 1;
    for (int i = 2; i < (P + 1); i++) {
      R = (R * (N1 + i)) / i;
    }
  }
  *out = R;
}

__device__ void IthCombination(int out[], int N, int P, int L) {
  // The out[p] can be calculated  using formula out[p] = out[p - 1] + L - K. note that out[p] is in 1-base indexing
  int P1 = P - 1;
  int R;
  int k = 0;
  for (int i = 0; i < P1; i++) {
    out[i] = 0;
    if (i > 0) {
      out[i] = out[i - 1];
    }
    while (k < L) {
      out[i] = out[i] + 1;
      BINOM(N - out[i], P - (i + 1), &R);
      k = k + R;
    }
    k = k - R;
  }
  out[P1] = out[P1 - 1] + L - k;
}
