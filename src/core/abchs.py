"""
Adaptive Boundary Constraint-Handling Scheme (ABCHS)
CRITICAL IMPLEMENTATION: Dynamic NFS computation based on PBEST feasibility

Key Fix:
- NFS computed from pbest CV values (CV <= 0), NOT current position feasibility
- NFS recomputed BEFORE each BCHM selection (dynamic, inside particle loop)
- Stage can transition from 1→2 mid-generation when first feasible pbest found
- Particle i+1 sees updated NFS from particle i's pbest update
"""

from typing import Optional, Tuple

import numpy as np

from src.benchmarks.cec2006 import CEC2006Problem
from src.core.boundary_handler import (
    METHOD_UNIF, CUSTOM_CENTROID, METHOD_MIRROR, METHOD_TOROIDAL
)
from src.core.pso import PSO
from src.core.velocity_strategy import VELOCITY_DEB, VELOCITY_RAB


class ABCHS(PSO):
    """
    Adaptive Boundary Constraint-Handling Scheme
    Extends PSO with adaptive method selection based on dynamic NFS
    """

    def __init__(self,
                 problem: CEC2006Problem,
                 population_size: int = 100,
                 constriction_k: float = 0.729,
                 c1: float = 1.49445,
                 c2: float = 1.49445,
                 max_evaluations: int = 500000,
                 random_seed: Optional[int] = None):
        """
        Initialize ABCHS

        Args:
            problem: CEC2006 problem instance
            population_size: Swarm size
            constriction_k: Constriction coefficient
            c1: Cognitive acceleration
            c2: Social acceleration
            max_evaluations: Maximum function evaluations
            random_seed: Random seed
        """
        # Initialize base PSO (position_method and velocity_strategy will be dynamic)
        super().__init__(
            problem=problem,
            population_size=population_size,
            constriction_k=constriction_k,
            c1=c1,
            c2=c2,
            max_evaluations=max_evaluations,
            position_method=METHOD_UNIF,  # Default, will be overridden
            velocity_strategy=VELOCITY_RAB,  # Default, will be overridden
            random_seed=random_seed
        )

        # ABCHS-specific state
        self.stage = 1  # Start in stage 1 (no feasible solutions yet)

        # Method pool for Stage 2: [Ran&RaB, Cen&DeB, Ref&DeB, Wra&RaB]
        self.method_pool = [
            (METHOD_UNIF, VELOCITY_RAB),  # Ran&RaB (bchm1)
            (CUSTOM_CENTROID, VELOCITY_DEB),  # Cen&DeB (bchm2)
            (METHOD_MIRROR, VELOCITY_DEB),  # Ref&DeB (bchm3)
            (METHOD_TOROIDAL, VELOCITY_RAB)  # Wra&RaB (bchm4)
        ]

        # Method probabilities (initially equal)
        self.probabilities = np.array([0.25, 0.25, 0.25, 0.25])
        self.epsilon = 0.01

        # Learning period
        self.learning_period = int(np.round(0.5 * problem.dimension)) + 2

        # Performance tracking (rsB = better, rsW = worse)
        self.rsB = np.zeros(4)
        self.rsW = np.zeros(4)

        # Transition tracking
        self.nfs_transitions = []  # List of (generation, particle_idx, nfs_before, nfs_after)
        self.stage_transitions = []  # List of (generation, particle_idx, stage_before, stage_after)
        self.methods_selected = []  # List of (generation, particle_idx, method_idx)

        # Track NFS at start of generation
        self.nfs_at_start = 0

        # Track last selected method ID for this generation (for measures collection)
        self.last_selected_method_id = None

    def _compute_nfs_dynamic(self) -> int:
        """
        CRITICAL: Compute NFS from PBEST feasibility (CV <= 0)
        This is called BEFORE each method selection to get current state

        Returns:
            Number of feasible pbests (CV <= 0)
        """
        return sum(1 for p in self.swarm if p.pbest_cv <= 0)

    def _select_method(self) -> Tuple[int, str]:
        """
        Select boundary handling method based on current stage

        Returns:
            (position_method, velocity_strategy)
        """
        if self.stage == 1:
            # Stage 1: Use Ran&RaB (promotes exploration)
            return self.method_pool[0]  # (METHOD_UNIF, VELOCITY_RAB)

        else:
            # Stage 2: Roulette wheel selection
            method_idx = self._roulette_wheel_selection()
            return self.method_pool[method_idx]

    def _roulette_wheel_selection(self) -> int:
        """
        Roulette wheel selection based on method probabilities

        Returns:
            Index of selected method (0-3)
        """
        cumsum = np.cumsum(self.probabilities)
        r = np.random.rand()

        for i in range(len(cumsum)):
            if r <= cumsum[i]:
                return i

        return len(cumsum) - 1  # Fallback

    def _update_probabilities(self):
        """
        Update method probabilities based on performance (Section 2 of Algorithm 3)
        Called every LP generations in Stage 2
        """
        # Compute quality scores
        S = np.zeros(4)
        for j in range(4):
            total = self.rsB[j] + self.rsW[j]
            if total > 0:
                S[j] = self.rsB[j] / total
            else:
                S[j] = 0

        # Normalize to probabilities
        sum_S = np.sum(S)
        if sum_S > 0:
            self.probabilities = S / sum_S + self.epsilon
        else:
            self.probabilities = np.full(4, 0.25)

        # Normalize again to ensure sum = 1
        self.probabilities = self.probabilities / np.sum(self.probabilities)

        # Reset counters
        self.rsB = np.zeros(4)
        self.rsW = np.zeros(4)

    def step(self):
        """
        Execute one generation following EXACT Algorithm 3 structure

        Section 1: Method selection during particle updates
        Section 2: Performance tracking at END of generation
        """
        # Increment generation at START
        self.generation += 1

        # Reset generation tracking for prob_infeas metric and gbest updates
        self.gen_violated_components = 0
        self.gen_total_components = 0
        self.gen_gbest_updates = 0  # Reset gbest update counter
        self.gen_particles_improved = 0  # Reset particle improvement counter

        # Store generation g-1 fitness for Algorithm 3 line 20
        prev_gen_fitness = [p.fitness for p in self.swarm]

        self.nfs_at_start = self._compute_nfs_dynamic()
        self.nfs_transitions = []
        self.stage_transitions = []
        self.methods_selected = []

        # Track repair info for Section 2
        repair_info = []  # List of (particle_idx, method_idx)

        for i, particle in enumerate(self.swarm):
            # CRITICAL: Compute NFS at start of particle processing
            # This must be done BEFORE any updates so we can track transitions
            nfs_before = self._compute_nfs_dynamic()

            # Store old state
            old_position = particle.position.copy()
            old_fitness = particle.fitness

            # 1. Update velocity (standard PSO)
            r1 = np.random.rand(self.problem.dimension)
            r2 = np.random.rand(self.problem.dimension)

            cognitive = self.c1 * r1 * (particle.pbest_position - particle.position)
            social = self.c2 * r2 * (self.gbest_position - particle.position)

            particle.velocity = self.k * (particle.velocity + cognitive + social)

            # 2. Update position
            particle.position = particle.position + particle.velocity

            # 3. Check bounds
            has_violations, violations = self.boundary_handler.check_bounds(particle.position)

            # Track boundary violations for prob_infeas metric
            # In PSO, all D dimensions are updated every generation
            self.gen_total_components += self.problem.dimension
            if has_violations:
                # Count how many dimensions violated bounds
                num_violated = np.sum(violations != 0)
                self.gen_violated_components += num_violated

            particle.was_repaired = False
            method_idx_used = None

            if has_violations:
                particle.was_repaired = True

                # Update NFS for method selection (should be same as nfs_before unless previous particle updated pbest)
                nfs_current = self._compute_nfs_dynamic()

                # Update stage based on CURRENT NFS
                stage_before = self.stage
                self.stage = 1 if nfs_current == 0 else 2

                # Track stage transition if it occurred
                if stage_before != self.stage:
                    self.stage_transitions.append((self.generation, i, stage_before, self.stage))

                # Section 1 of Algorithm 3: Select method based on CURRENT stage
                position_method, velocity_strategy = self._select_method()

                # Track which method from pool was selected
                if self.stage == 2:
                    for idx, method in enumerate(self.method_pool):
                        if method == (position_method, velocity_strategy):
                            method_idx_used = idx
                            self.methods_selected.append((self.generation, i, idx))
                            self.last_selected_method_id = idx
                            break
                else:
                    method_idx_used = 0  # Stage 1 always uses method 0 (Ran&RaB)
                    self.methods_selected.append((self.generation, i, 0))
                    self.last_selected_method_id = 0

                # Save repair info for Section 2 tracking
                repair_info.append((i, method_idx_used))

                particle.repair_method_used = position_method

                # Apply position correction
                particle.position = self.boundary_handler.apply_method(
                    particle.position,
                    position_method,
                    target=old_position,
                    best=self.gbest_position,
                    pbest_population=self._get_pbest_population()
                )

                # Apply velocity update
                from src.core.velocity_strategy import VelocityUpdater
                particle.velocity = VelocityUpdater.update_velocity(
                    particle.velocity,
                    violations,
                    velocity_strategy
                )

            # 4. Evaluate new position
            self._evaluate_particle(particle)

            # 5. Update personal best
            old_pbest_cv = particle.pbest_cv
            improved = self._update_personal_best(particle)
            if improved:
                self.gen_particles_improved += 1

            # 6. Update global best IMMEDIATELY (asynchronous update!)
            # Particle i+1 will see the improved gbest from particle i in THIS generation
            from src.core.constraint_handler import DebsRules
            if DebsRules.is_better(
                    particle.pbest_fitness, particle.pbest_cv,
                    self.gbest_fitness, self.gbest_cv,
                    minimize=True
            ):
                self.gbest_position = particle.pbest_position.copy()
                self.gbest_fitness = particle.pbest_fitness
                self.gbest_cv = particle.pbest_cv
                self.gbest_is_feasible = particle.pbest_cv <= 0

            # Track NFS transition if pbest feasibility changed
            nfs_after = self._compute_nfs_dynamic()
            if nfs_before != nfs_after:
                self.nfs_transitions.append((self.generation, i, nfs_before, nfs_after))

        # ========================================================================
        # SECTION 2 of Algorithm 3 (lines 9-29): AT END OF GENERATION
        # ========================================================================
        # Update NFS at end of generation
        self.nfs = self._compute_nfs_dynamic()

        # Algorithm 3, line 10: if NFS > 0 then
        if self.nfs > 0:
            # Algorithm 3, line 11: if (g % LP) = 0 then
            if (self.generation % self.learning_period) == 0:
                # Lines 12-15: Update probabilities and RESET counters
                self._update_probabilities()
            else:
                # Lines 16-23 (ELSE branch): Update rsB/rsW counters
                for particle_idx, method_idx in repair_info:
                    # Line 20: if f(X_i^g) <= f(X_i^(g-1)) then
                    current_fitness = self.swarm[particle_idx].fitness
                    previous_fitness = prev_gen_fitness[particle_idx]

                    if current_fitness <= previous_fitness:
                        # Line 21: rsB_j = rsB_j + 1
                        self.rsB[method_idx] += 1
                    else:
                        # Line 23: rsW_j = rsW_j + 1
                        self.rsW[method_idx] += 1

    def get_state(self) -> dict:
        """Get current state including ABCHS-specific info"""
        state = super().get_state()

        # Add ABCHS-specific state
        state.update({
            'stage': self.stage,
            'nfs_at_start': self.nfs_at_start,
            'nfs_transitions': self.nfs_transitions.copy(),
            'stage_transitions': self.stage_transitions.copy(),
            'probabilities': self.probabilities.copy(),
            'rsB': self.rsB.copy(),
            'rsW': self.rsW.copy(),
            'learning_period': self.learning_period,
            'methods_selected': self.methods_selected.copy()
        })

        return state
