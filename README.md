# Primal-Dual-Techniques-for-Distributed-Multitask-Learning  
== Overview  

Distributed learning techniques allow a collection of intelligent agents to cooperatively solve optimisation and learning problems while relying on
limited interactions and ensuring privacy, communication efficiency and robustness.  

Most strategies are designed for so called "singletask" problems, where agents solve a single, common problem and agree on a single, common
model. Such techniques are appropriate in homogenous environments, but fail when agents are heterogeneous, and instead wish to learn distinct
but related models. To this end, recent works have developed techniques for distributed learning under subspace constraints, where local models
are distinct, but lie on a lower dimensional subspace. While effective in the presence of noise, the resulting algorithms exhibit a small but
significant bias. This project will develop novel algorithms based on primal-dual algorithms which remove this bias and are expected to yield
superior performance in the low-noise regime.  

== Requirements  
- Proficiency with optimization and machine learning
- Experience in implementing learning algorithms in python and relevant libraries
  
== References
- https://ieeexplore.ieee.org/abstract/document/9084370
- https://arxiv.org/abs/2210.13767
