"""
Velocity Update Strategies for Boundary Constraint Handling
"""

from typing import Literal

import numpy as np

# Strategy constants
VELOCITY_DEB = "DeB"  # Deterministic Back
VELOCITY_RAB = "RaB"  # Random Back


class VelocityUpdater:
    """
    Velocity update strategies for boundary constraint handling
    """

    @staticmethod
    def update_velocity(velocity: np.ndarray, violations: np.ndarray,
                        strategy: Literal["DeB", "RaB"]) -> np.ndarray:
        """
        Update velocity for violated dimensions

        Args:
            velocity: Current velocity vector
            violations: Violations mask (0 = ok, -1 = lower, 1 = upper)
            strategy: "DeB" or "RaB"

        Returns:
            Updated velocity vector
        """
        updated_velocity = velocity.copy()

        if strategy == VELOCITY_DEB:
            # Deterministic Back: V_j^c = -0.5 * V_j
            updated_velocity[violations != 0] *= -0.5

        elif strategy == VELOCITY_RAB:
            # Random Back: V_j^c = -λ * V_j where λ ~ Uniform[0,1]
            for i in range(len(velocity)):
                if violations[i] != 0:
                    lambda_val = np.random.uniform(0, 1)
                    updated_velocity[i] *= -lambda_val

        else:
            raise ValueError(f"Unknown velocity strategy: {strategy}")

        return updated_velocity

    @staticmethod
    def apply_deb(velocity: np.ndarray, violations: np.ndarray) -> np.ndarray:
        """
        Apply Deterministic Back strategy

        Args:
            velocity: Current velocity vector
            violations: Violations mask

        Returns:
            Updated velocity: -0.5 * velocity for violated dimensions
        """
        return VelocityUpdater.update_velocity(velocity, violations, VELOCITY_DEB)

    @staticmethod
    def apply_rab(velocity: np.ndarray, violations: np.ndarray) -> np.ndarray:
        """
        Apply Random Back strategy

        Args:
            velocity: Current velocity vector
            violations: Violations mask

        Returns:
            Updated velocity: -λ * velocity for violated dimensions (λ ~ U[0,1])
        """
        return VelocityUpdater.update_velocity(velocity, violations, VELOCITY_RAB)
