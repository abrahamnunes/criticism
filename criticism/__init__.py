# -*- coding: utf-8 -*-

"""Top-level package for criticism."""

__author__ = """Abraham Nunes"""
__email__ = 'nunes@dal.ca'
__version__ = '0.1.0'

from criticism import metrics
from criticism.criticism import NullClassifierSim
from criticism.criticism import MultiNullClassifierSim
from criticism.criticism import TestStatisticSim
from criticism.criticism import EmpiricalDistribution
from criticism.criticism import SimulationScorer

__all__ = [
    'metrics',
    'NullClassifierSim',
    'MultiNullClassifierSim',
    'TestStatisticSim',
    'EmpiricalDistribution',
    'SimulationScorer'
]
