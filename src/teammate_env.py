# src/teammate_env.py

"""
Shim module so tests can do:

    from src.teammate_env import IndependentCascadeEnvironment
"""

from .tc_wrapper import IndependentCascadeEnvironment

__all__ = ["IndependentCascadeEnvironment"]
