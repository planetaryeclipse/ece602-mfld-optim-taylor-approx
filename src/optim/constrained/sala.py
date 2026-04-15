import torch

from optim.constrained.ralm import AugmentedLagrangian

# implements the Safeguarded Augmented Lagrangian Algorithm (SALA) described in "Constraint Qualifications and Strong
# Global Convergence Properties of an Augmented Lagrangian Method on Riemannian Manifolds"
# NOTE: the definition of the augmented lagrangian in this paper is the same as that used to define RALM