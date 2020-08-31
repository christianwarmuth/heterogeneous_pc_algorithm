# causal_inference_research
## Heterogeneous Computing for Constraint-Based Causal Structure Learning
Seminar: Causal Inference - Theory and Applications in Enterprise Computing

## Abstract
Learning cause-effect relationships is a central goal in many research areas. Various techniques and algorithms exist to learn causal structures solely based on observational data since interventions are often infeasible for real-world applications. The PC algorithm is a constraint-based causal structure learning algorithm to learn the underlying causal structure of observed variables by performing conditional independence tests (CI-tests). Motivated by the steady advances in hardware technology, purely CPU- or GPU-based parallel implementations of the PC algorithm such as pcalg and cuPC do not harness the computational power of all processing units available in a heterogeneous setting.

In this paper, we study heterogeneous CPU-GPU computing for constraint-based causal structure learning. We evaluate two state-of-the-art implementations of the PC algorithm (i.e., pcalg and cuPC) and derive potential performance improvements by utilizing the available heterogeneous setup. We propose a general approach for heterogeneous constraint-based causal structure learning using static workload-partitioning. The core idea is to perform the CI-tests simultaneously on both CPU and GPU and merge the results afterwards. We provide a reference implementation based on pcalg and cuPC for our heterogeneous approach. Our experimental evaluation shows that for large problem sizes the additional overhead of the heterogeneous setting becomes negligible and we achieve comparable performance to modern PC algorithm implementations while utilizing the available resources of both CPU and GPU.

## Build commands:
```make cupc_heterogeneous```

## Run-scripts:
```cd ./scripts```
```Rscript run_cupc_heterogeneous.R```

## Based on the implementation [pcalg](https://github.com/cran/pcalg) and [cuPC](https://github.com/LIS-Laboratory/cupc)

## Contributors:
-  [Tobias Maltenberger](https://github.com/maltenbergert)
-  [Christian Warmuth](https://github.com/christianwarmuth)
