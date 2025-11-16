"""
Deb's Rules for Constrained Optimization
Comparison rules for solutions in presence of constraints
"""

import numpy as np


class DebsRules:
    """
    Deb's constraint handling rules for comparing solutions:

    Rule 1: Between two feasible solutions, prefer the one with better fitness
    Rule 2: Between feasible and infeasible, prefer feasible
    Rule 3: Between two infeasible solutions, prefer the one with less constraint violation
    """

    @staticmethod
    def is_better(fitness_a: float, cv_a: float,
                  fitness_b: float, cv_b: float,
                  minimize: bool = True) -> bool:
        """
        Check if solution A is better than solution B using Deb's rules

        Args:
            fitness_a: Fitness value of solution A
            cv_a: Constraint violation of solution A (0 if feasible)
            fitness_b: Fitness value of solution B
            cv_b: Constraint violation of solution B (0 if feasible)
            minimize: True if minimization problem, False if maximization

        Returns:
            True if A is better than B, False otherwise
        """
        feasible_a = cv_a <= 0
        feasible_b = cv_b <= 0

        # Rule 2: Feasible vs Infeasible
        if feasible_a and not feasible_b:
            return True
        if not feasible_a and feasible_b:
            return False

        # Rule 1: Both feasible - compare fitness
        if feasible_a and feasible_b:
            if minimize:
                return fitness_a < fitness_b
            else:
                return fitness_a > fitness_b

        # Rule 3: Both infeasible - compare constraint violations
        return cv_a < cv_b

    @staticmethod
    def is_better_or_equal(fitness_a: float, cv_a: float,
                           fitness_b: float, cv_b: float,
                           minimize: bool = True) -> bool:
        """
        Check if solution A is better than or equal to solution B

        Returns:
            True if A is better than or equal to B, False otherwise
        """
        if DebsRules.is_better(fitness_a, cv_a, fitness_b, cv_b, minimize):
            return True

        # Check equality
        feasible_a = cv_a <= 0
        feasible_b = cv_b <= 0

        if feasible_a and feasible_b:
            return np.isclose(fitness_a, fitness_b)

        if not feasible_a and not feasible_b:
            return np.isclose(cv_a, cv_b)

        return False

    @staticmethod
    def compare(fitness_a: float, cv_a: float,
                fitness_b: float, cv_b: float,
                minimize: bool = True) -> int:
        """
        Compare two solutions using Deb's rules

        Returns:
            1 if A is better than B
            -1 if B is better than A
            0 if they are equal
        """
        if DebsRules.is_better(fitness_a, cv_a, fitness_b, cv_b, minimize):
            return 1
        elif DebsRules.is_better(fitness_b, cv_b, fitness_a, cv_a, minimize):
            return -1
        else:
            return 0

    @staticmethod
    def select_best(solutions: list, minimize: bool = True) -> int:
        """
        Select the best solution from a list using Deb's rules

        Args:
            solutions: List of tuples (fitness, cv, index)
            minimize: True if minimization problem

        Returns:
            Index of the best solution
        """
        if not solutions:
            raise ValueError("Empty solutions list")

        best_idx = 0
        best_fitness, best_cv = solutions[0][0], solutions[0][1]

        for i in range(1, len(solutions)):
            fitness, cv = solutions[i][0], solutions[i][1]
            if DebsRules.is_better(fitness, cv, best_fitness, best_cv, minimize):
                best_idx = i
                best_fitness, best_cv = fitness, cv

        return best_idx

    @staticmethod
    def rank_solutions(solutions: list, minimize: bool = True) -> list:
        """
        Rank solutions using Deb's rules

        Args:
            solutions: List of tuples (fitness, cv, original_index)
            minimize: True if minimization problem

        Returns:
            List of original indices sorted from best to worst
        """
        n = len(solutions)
        if n == 0:
            return []

        # Sort using Deb's rules
        sorted_solutions = sorted(
            solutions,
            key=lambda s: (
                s[1] > 0,  # Feasible first (False < True)
                s[1] if s[1] > 0 else (s[0] if minimize else -s[0])  # Then by CV or fitness
            )
        )

        return [s[2] for s in sorted_solutions]
