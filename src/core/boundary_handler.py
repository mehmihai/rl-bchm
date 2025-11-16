"""
Boundary Constraint-Handling Methods (BCHMs)
EXACT implementations of all position handling methods
Including simple methods, complex methods (Centroid, Evolutionary), and vectorial correction
"""

from typing import Tuple, List, Optional

import numpy as np

# Method constants
METHOD_SATURATION = 0  # Boundary
METHOD_MIDPOINT_TARGET = 1
METHOD_MIDPOINT_BEST = 2
METHOD_UNIF = 3  # Random/Uniform
METHOD_MIRROR = 5  # Reflection
METHOD_TOROIDAL = 6  # Wrapping
METHOD_EXPC_TARGET = 8
METHOD_EXPC_BEST = 9
METHOD_VECTOR_TARGET = 11
METHOD_VECTOR_BEST = 12
METHOD_DISMISS = 13
CUSTOM_CENTROID = 14
CUSTOM_EVOLUTIONARY = 15


class BoundaryHandler:
    """
    Comprehensive boundary constraint-handling methods
    EXACT implementations from correction_handler_A.py
    """

    def __init__(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        """
        Initialize boundary handler

        Args:
            lower_bounds: Lower bounds for each dimension
            upper_bounds: Upper bounds for each dimension
        """
        self.lower = np.array(lower_bounds)
        self.upper = np.array(upper_bounds)
        self.dimension = len(lower_bounds)

    def check_bounds(self, position: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Check if position violates bounds

        Returns:
            (has_violations, violations_mask) where violations_mask[i] indicates:
                -1: component i is below lower bound
                0: component i is within bounds
                1: component i is above upper bound
        """
        violations = np.zeros(self.dimension, dtype=int)
        violations[position < self.lower] = -1
        violations[position > self.upper] = 1
        has_violations = np.any(violations != 0)
        return has_violations, violations

    def apply_method(self, position: np.ndarray, method_id: int,
                     target: Optional[np.ndarray] = None,
                     best: Optional[np.ndarray] = None,
                     pbest_population: Optional[List[Tuple[np.ndarray, float, float]]] = None) -> np.ndarray:
        """
        Apply boundary handling method to position

        Args:
            position: Current position (may have violations)
            method_id: Method constant
            target: Target/old position (for methods that need it)
            best: Global best position (for methods that need it)
            pbest_population: List of (pbest_position, pbest_fitness, pbest_cv) for Centroid

        Returns:
            Corrected position
        """
        has_violations, violations = self.check_bounds(position)

        if not has_violations:
            return position.copy()

        # Handle complex methods that operate on entire vector
        if method_id == CUSTOM_CENTROID:
            return self._centroid_method(position, pbest_population, violations)
        elif method_id == METHOD_VECTOR_TARGET:
            if target is None:
                target = position.copy()
            return self._vectorial_correction(position, target, violations)[0]
        elif method_id == METHOD_VECTOR_BEST:
            if best is None:
                raise ValueError("Vector_Best requires global best position")
            return self._vectorial_correction(position, best, violations)[0]
        elif method_id == METHOD_DISMISS:
            # Dismiss: Replace entire vector with last feasible parent (target)
            # FIXED: Verify target has NO BOUNDARY VIOLATIONS (not functional constraints!)
            if target is None:
                # Fallback: generate random feasible solution
                return np.random.uniform(self.lower, self.upper)

            # Check if target has BOUNDARY violations
            target_violations = np.zeros(self.dimension, dtype=int)
            target_violations[target < self.lower] = -1
            target_violations[target > self.upper] = 1

            if np.any(target_violations != 0):
                # Target has boundary violations, use random feasible solution
                return np.random.uniform(self.lower, self.upper)

            # Target is within bounds, use it
            return target.copy()

        # Handle component-wise methods
        corrected = position.copy()
        for i in range(self.dimension):
            if violations[i] != 0:
                corrected[i] = self._apply_component_method(
                    position[i], i, violations[i], method_id,
                    target[i] if target is not None else None,
                    best[i] if best is not None else None
                )

        return corrected

    def _apply_component_method(self, component: float, dim: int, violation: int,
                                method_id: int, target_comp: Optional[float],
                                best_comp: Optional[float]) -> float:
        """Apply boundary handling to a single component"""

        lower = self.lower[dim]
        upper = self.upper[dim]

        if method_id == METHOD_SATURATION:
            # Boundary: Reset to violated bound
            return lower if violation == -1 else upper

        elif method_id == METHOD_MIDPOINT_TARGET:
            # Midpoint between bound and target
            if target_comp is None:
                target_comp = component
            return (lower + target_comp) / 2 if violation == -1 else (upper + target_comp) / 2

        elif method_id == METHOD_MIDPOINT_BEST:
            # Midpoint between bound and best
            if best_comp is None:
                raise ValueError("Midpoint_Best requires global best")
            return (lower + best_comp) / 2 if violation == -1 else (upper + best_comp) / 2

        elif method_id == METHOD_UNIF:
            # Random/Uniform: Random value within bounds
            return np.random.uniform(lower, upper)

        elif method_id == METHOD_MIRROR:
            # Reflection: Mirror from violated bound
            if violation == -1:
                return 2 * lower - component
            else:
                return 2 * upper - component

        elif method_id == METHOD_TOROIDAL:
            # Wrapping: Toroidal wrapping with iterative verification
            # FIXED: Re-apply wrapping until result is within bounds (no iteration limit)
            corrected = component

            while True:
                # Check if within bounds
                if lower <= corrected <= upper:
                    return corrected

                # Apply wrapping based on which bound is violated
                if corrected < lower:
                    corrected = upper - (lower - corrected)
                else:  # corrected > upper
                    corrected = lower + (corrected - upper)

        elif method_id == METHOD_EXPC_TARGET:
            # Exponential correction towards target - EXACT implementation
            if target_comp is None:
                target_comp = component

            if violation == -1:
                # Lower bound violation
                diff = lower - target_comp
                if diff >= 0:
                    # target is below or at lower bound
                    return lower
                r = np.random.uniform(0, 1)
                exp_term = np.exp(diff) - 1
                if exp_term == 0:
                    return lower
                return lower - np.log(1 + r * exp_term)
            else:
                # Upper bound violation
                diff = target_comp - upper
                if diff >= 0:
                    # target is above or at upper bound
                    return upper
                r = np.random.uniform(0, 1)
                exp_term = np.exp(diff) - 1
                if exp_term == 0:
                    return upper
                return upper + np.log(1 + (1 - r) * exp_term)

        elif method_id == METHOD_EXPC_BEST:
            # Exponential correction towards best - EXACT implementation
            if best_comp is None:
                raise ValueError("ExpC_Best requires global best")

            if violation == -1:
                # Lower bound violation
                diff = lower - best_comp
                if diff >= 0:
                    return lower
                r = np.random.uniform(0, 1)
                exp_term = np.exp(diff) - 1
                if exp_term == 0:
                    return lower
                return lower - np.log(1 + r * exp_term)
            else:
                # Upper bound violation
                diff = best_comp - upper
                if diff >= 0:
                    return upper
                r = np.random.uniform(0, 1)
                exp_term = np.exp(diff) - 1
                if exp_term == 0:
                    return upper
                return upper + np.log(1 + (1 - r) * exp_term)

        elif method_id == CUSTOM_EVOLUTIONARY:
            # Evolutionary: Random convex combination between bound and best
            # Eq. 14: x_j^c = α × l_j + (1 - α) × X_j^best  if X_j < l_j
            #         x_j^c = β × u_j + (1 - β) × X_j^best  if X_j > u_j
            if best_comp is None:
                raise ValueError("Evolutionary method requires global best")

            if violation == -1:
                # Lower bound violation
                alpha = np.random.uniform(0, 1)
                return alpha * lower + (1 - alpha) * best_comp
            else:
                # Upper bound violation
                beta = np.random.uniform(0, 1)
                return beta * upper + (1 - beta) * best_comp

        else:
            raise ValueError(f"Unknown method ID: {method_id}")

    def _vectorial_correction(self, trial_vector: np.ndarray, reference: np.ndarray,
                              violations: np.ndarray) -> Tuple[np.ndarray, List[int], float]:
        """
        Vectorial correction method - EXACT from correction_handler_A.py

        Args:
            trial_vector: Vector with violations
            reference: Reference vector (target or best)
            violations: Violations mask

        Returns:
            (corrected_vector, repaired_indices, gamma)
        """
        alpha = np.zeros(self.dimension)
        repaired = [False] * self.dimension

        # Compute alpha for each dimension
        for j in range(self.dimension):
            if violations[j] == -1:  # Below lower bound
                if trial_vector[j] != reference[j]:
                    alpha[j] = (self.lower[j] - reference[j]) / (trial_vector[j] - reference[j])
                else:
                    alpha[j] = np.inf
                repaired[j] = True
            elif violations[j] == 1:  # Above upper bound
                if trial_vector[j] != reference[j]:
                    alpha[j] = (self.upper[j] - reference[j]) / (trial_vector[j] - reference[j])
                else:
                    alpha[j] = np.inf
                repaired[j] = True
            else:
                alpha[j] = np.inf

        # Find minimum alpha (gamma)
        valid_alphas = alpha[alpha != np.inf]
        if len(valid_alphas) == 0:
            gamma = 0.5  # Default if no valid alphas
        else:
            gamma = np.min(valid_alphas)

        # Compute corrected vector
        corrected = gamma * trial_vector + (1 - gamma) * reference

        # Project back to bounds if still violated
        corrected = np.clip(corrected, self.lower, self.upper)

        repaired_indices = [i for i in range(self.dimension) if repaired[i]]

        return corrected, repaired_indices, gamma

    def _centroid_method(self, position: np.ndarray,
                         pbest_population: Optional[List[Tuple[np.ndarray, float, float]]],
                         violations: np.ndarray) -> np.ndarray:
        """
        Centroid method - FULL implementation from paper

        Args:
            position: Position with violations
            pbest_population: List of (pbest_position, pbest_fitness, pbest_cv)
            violations: Violations mask

        Returns:
            Corrected position
        """
        # Step 1: Check if all components are feasible
        if not np.any(violations != 0):
            return position.copy()

        # Step 2: Select wp based on AFS (Amount of Feasible Solutions)
        if pbest_population is None or len(pbest_population) == 0:
            # Fallback: use random position within bounds
            wp = np.random.uniform(self.lower, self.upper)
        else:
            # Compute SFS (Set of Feasible Solutions) and SIS (Set of Infeasible Solutions)
            # based on PBEST population and FUNCTIONAL constraint feasibility
            SFS = [pbest for pbest, _, cv in pbest_population if cv <= 0]
            SIS = [pbest for pbest, _, cv in pbest_population if cv > 0]

            AFS = len(SFS)

            if AFS > 0 and np.random.rand() > 0.5:
                # Select random feasible pbest
                wp = SFS[np.random.randint(0, len(SFS))].copy()
            elif len(SIS) > 0:
                # Select best infeasible pbest (lowest CV)
                sis_cvs = [cv for _, _, cv in pbest_population if cv > 0]
                best_sis_idx = np.argmin(sis_cvs)
                wp = SIS[best_sis_idx].copy()
            else:
                # Fallback
                wp = pbest_population[0][0].copy()

        # Step 3: Generate K random vectors (K=1)
        K = 1
        wr1 = position.copy()

        # Replace invalid components with random values
        for i in range(self.dimension):
            if violations[i] != 0:
                wr1[i] = np.random.uniform(self.lower[i], self.upper[i])

        # Step 4: Compute centroid
        centroid = (wp + wr1) / 2.0

        # Ensure within bounds
        centroid = np.clip(centroid, self.lower, self.upper)

        return centroid


# Method name mappings
METHOD_NAMES = {
    METHOD_SATURATION: "Boundary",
    METHOD_MIDPOINT_TARGET: "Midpoint_Target",
    METHOD_MIDPOINT_BEST: "Midpoint_Best",
    METHOD_UNIF: "Random",
    METHOD_MIRROR: "Reflection",
    METHOD_TOROIDAL: "Wrapping",
    METHOD_EXPC_TARGET: "ExpC_Target",
    METHOD_EXPC_BEST: "ExpC_Best",
    METHOD_VECTOR_TARGET: "Vector_Target",
    METHOD_VECTOR_BEST: "Vector_Best",
    METHOD_DISMISS: "Dismiss",
    CUSTOM_CENTROID: "Centroid",
    CUSTOM_EVOLUTIONARY: "Evolutionary"
}


def get_method_name(method_id: int) -> str:
    """Get human-readable method name"""
    return METHOD_NAMES.get(method_id, f"Unknown({method_id})")
