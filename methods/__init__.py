"""
Experimental Meta-Analysis Methods Package
==========================================
Contains experimental methods across 5 modules:

    - Part 1: Adaptive, Mixture Models, Kernel-based, Information-theoretic
    - Part 2: Robust and Resistant Methods
    - Part 3: Bayesian and Information-theoretic
    - Part 4: Machine Learning and Ensemble
    - Part 5: Penalized Likelihood and Regularization
"""

from .experimental_methods_part1 import get_part1_methods
from .experimental_methods_part2 import get_part2_methods
from .experimental_methods_part3 import get_part3_methods
from .experimental_methods_part4 import get_part4_methods
from .experimental_methods_part5 import get_part5_methods

__all__ = [
    "get_part1_methods",
    "get_part2_methods",
    "get_part3_methods",
    "get_part4_methods",
    "get_part5_methods",
]


def get_all_experimental_methods():
    """Get all experimental methods from all parts"""
    methods = []
    methods.extend(get_part1_methods())
    methods.extend(get_part2_methods())
    methods.extend(get_part3_methods())
    methods.extend(get_part4_methods())
    methods.extend(get_part5_methods())
    return methods
