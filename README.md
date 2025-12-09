Topology-Informed Kernel Density Estimation
Code Repository for paper titled: “Unsupervised Learning of Density Estimates with Topological Optimization”

For main experiments, run submit_main_simulations.sb on HPC which calls R_5_main_simulation.py with various configurations (datasets and seeds). Then run R_6_analyze.py to generate tables of KLD/EMD values for all methods compared.

For ablation study (appendix B) results, run submit_ablation_study.sb which calls R_7_ablation_study.py with different configurations. Then run R_8_analyze_ablation.py to print out result tables.

For hyperparameter sensitivity study (appendix A), run submit_hyperpar_sensitivity_study.sb which calls R_9_hyperparameter_sensitivity.py with various configurations. Then run R_10_analyze_sensitivity.py to generate plots. 

For figures in the paper, run figures_for_paper.ipynb.

Please cite:
@article{Tanweer2025,
  title   = {Unsupervised Learning of Density Estimates with Topological Optimization},
  author  = {Tanweer, Sunia \& Khasawneh, Firas A.},
  journal = {arXiv preprint},
  year    = {2025}
}
