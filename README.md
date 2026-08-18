# Systems Consolidation

This project contains computational neural-network models for studying systems memory consolidation across hippocampal and cortical regions. The simulations explore how neural activity, excitability, synaptic plasticity, and representational drift influence learning and memory recall in two- and three-region rate-based networks.

## Two-region model

`app/TwoRegion_np.py` represents the hippocampus (HPC/MTL) and anterior cingulate cortex (ACC/CTX) as interacting populations of firing-rate neurons. A memory is initially encoded through activity-dependent Hebbian plasticity in recurrent and long-range synapses, after which repeated offline reactivation models the gradual consolidation of the memory into cortex. The model also introduces newly excitable neuronal cohorts over successive days, allowing it to examine engram allocation, intrinsic excitability, representational drift, and changes in memory recall. Optional perturbations—including regional plasticity blockade, intrinsic-plasticity blockade, and selective erasure of potentiated synapses—can be used to test which mechanisms support consolidation at different time points.
