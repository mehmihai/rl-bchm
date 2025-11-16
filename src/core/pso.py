"""
Particle Swarm Optimization Engine with Hybrid BCHM Support
Canonical PSO with constriction coefficient
"""

from typing import Optional, Tuple, List

import numpy as np

from src.benchmarks.cec2006 import CEC2006Problem
from src.core.boundary_handler import BoundaryHandler
from src.core.constraint_handler import DebsRules
from src.core.velocity_strategy import VelocityUpdater


class Particle:
    """
    Particle in PSO swarm

    PARTICLE STATE COMPONENTS (per paper's Algorithm 1):
    ====================================================
    Each particle i maintains:

    1. Current state:
       - position x[i]: Current position in D-dimensional search space
       - velocity v[i]: Current velocity (direction and magnitude of movement)
       - fitness f(x[i]): Objective function value at current position
       - cv: Constraint violation (sum of violated constraints)

    2. Personal best (pbest):
       - pbest_position: Best position this particle has found so far
       - pbest_fitness: Fitness at pbest position
       - pbest_cv: Constraint violation at pbest position
       - Used in velocity update: v = k*(v + c1*r1*(pbest - x) + ...)

    3. Boundary handling tracking:
       - was_repaired: Flag indicating if position was repaired this generation
       - repair_method_used: Which BCHM method was used for repair
       - Used by ABCHS to track repair performance

    INITIALIZATION:
    ---------------
    - Position: Random uniform within [lower_bounds, upper_bounds]
    - Velocity: Random uniform in [-(ub-lb), (ub-lb)] * 0.5 (smaller initial velocities)
    - Personal best: Initially set to initial position
    """

    def __init__(self, dimension: int, lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        """
        Initialize particle with random position and velocity

        Args:
            dimension: Problem dimensionality (D)
            lower_bounds: Lower bounds for each dimension
            upper_bounds: Upper bounds for each dimension
        """
        self.dimension = dimension

        # Current position x[i] - random initialization within bounds
        self.position = np.random.uniform(lower_bounds, upper_bounds)

        # Current velocity v[i] - random initialization with damping
        self.velocity = np.random.uniform(
            -(upper_bounds - lower_bounds),
            (upper_bounds - lower_bounds)
        ) * 0.5  # Initialize with smaller velocities for stability

        # Current state (updated after evaluation)
        self.fitness = np.inf
        self.cv = np.inf  # Constraint violation (0 if feasible, >0 if infeasible)
        self.is_feasible = False

        # Personal best (pbest) - best position this particle has visited
        self.pbest_position = self.position.copy()
        self.pbest_fitness = np.inf
        self.pbest_cv = np.inf
        self.pbest_is_feasible = False

        # Boundary handling tracking (for ABCHS performance analysis)
        self.was_repaired = False  # True if position was repaired this generation
        self.repair_method_used = None  # Which BCHM method was used


class PSO:
    """
    Canonical Particle Swarm Optimization with Hybrid BCHM

    IMPLEMENTATION OVERVIEW:
    ========================
    This implements the standard PSO algorithm from the paper's Algorithm 1, with support
    for Boundary Constraint-Handling Methods (BCHM).

    ALGORITHM STRUCTURE (per paper's Algorithm 1):
    ------------------------------------------------
    1. Initialize population randomly within bounds
    2. Evaluate all particles
    3. Set personal bests (pbest) and global best (gbest)
    4. REPEAT until max evaluations:
        For each particle i:
            a) Update velocity: v[i] = k*(v[i] + c1*r1*(pbest[i]-x[i]) + c2*r2*(gbest-x[i]))
            b) Update position: x[i] = x[i] + v[i]
            c) If position violates bounds:
                - Apply position repair method (Random, Centroid, Reflection, etc.)
                - Apply velocity update strategy (DeB, RaB, etc.)
            d) Evaluate particle at new position
            e) Update pbest if current position is better (using Deb's rules)
            f) Update gbest IMMEDIATELY if pbest is better (asynchronous update!)

    KEY DESIGN DECISIONS:
    ---------------------
    - Constriction coefficient formulation: k=0.729, c1=c2=1.49445
    - Deb's rules for constraint handling (feasible > infeasible)
    - Vectorized operations: ALL components updated before bounds check
    - BCHM applied per-particle after position update
    - ASYNCHRONOUS gbest update: particles i+1,...,N benefit from particle i's discoveries
      within the same generation (not synchronous where all use same gbest per generation)

    ABCHS EXTENSION:
    ----------------
    ABCHS (Adaptive Boundary Constraint-Handling Scheme) extends this base PSO by:
    1. Overriding step() to adaptively select BCHM methods
    2. Using two-stage approach: Stage 1 (exploration), Stage 2 (adaptive learning)
    3. Tracking repair performance (rsB/rsW) to update method probabilities
    4. Selecting methods via roulette wheel based on learned probabilities

    Uses constriction coefficient formulation with Deb's rules for constraint handling.
    """

    def __init__(self,
                 problem: CEC2006Problem,
                 population_size: int = 100,
                 constriction_k: float = 0.729,
                 c1: float = 1.49445,
                 c2: float = 1.49445,
                 max_evaluations: int = 500000,
                 position_method: int = 0,
                 velocity_strategy: str = "DeB",
                 random_seed: Optional[int] = None):
        """
        Initialize PSO with hybrid boundary constraint-handling

        Args:
            problem: CEC2006 problem instance
            population_size: Swarm size
            constriction_k: Constriction coefficient (typically 0.729)
            c1: Cognitive acceleration coefficient
            c2: Social acceleration coefficient
            max_evaluations: Maximum function evaluations
            position_method: Position handling method ID
            velocity_strategy: "DeB" or "RaB"
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.problem = problem
        self.population_size = population_size
        self.k = constriction_k
        self.c1 = c1
        self.c2 = c2
        self.max_evaluations = max_evaluations

        # Boundary constraint-handling configuration
        self.position_method = position_method
        self.velocity_strategy = velocity_strategy

        # Initialize boundary handler
        self.boundary_handler = BoundaryHandler(
            problem.lower_bounds,
            problem.upper_bounds
        )

        # Initialize swarm
        self.swarm = [
            Particle(problem.dimension, problem.lower_bounds, problem.upper_bounds)
            for _ in range(population_size)
        ]

        # Global best
        self.gbest_position = None
        self.gbest_fitness = np.inf
        self.gbest_cv = np.inf
        self.gbest_is_feasible = False

        # Tracking
        self.nfe = 0  # Number of function evaluations
        self.generation = 0
        self.nfs = 0  # Number of feasible solutions (pbest-based)

        # Boundary violation tracking (for prob_infeas metric)
        self.gen_violated_components = 0  # Components that violated bounds this generation
        self.gen_total_components = 0  # Total components updated this generation

        # Gbest update tracking (for asynchronous PSO)
        self.gen_gbest_updates = 0  # Number of times gbest was updated this generation

        # Particle improvement tracking
        self.gen_particles_improved = 0  # Number of particles that improved pbest this generation

        # Initialize population
        self._initialize_population()

    def _initialize_population(self):
        """Evaluate initial population and set personal/global bests"""
        for particle in self.swarm:
            self._evaluate_particle(particle)
            self._update_personal_best(particle)

        self._update_global_best()
        self._compute_nfs()

    def _evaluate_particle(self, particle: Particle):
        """Evaluate particle fitness and constraints"""
        fitness, cv, is_feasible = self.problem.evaluate(particle.position)
        particle.fitness = fitness
        particle.cv = cv
        particle.is_feasible = is_feasible
        self.nfe += 1

    def _update_personal_best(self, particle: Particle) -> bool:
        """
        Update particle's personal best using Deb's rules

        Returns:
            True if pbest was updated, False otherwise
        """
        if DebsRules.is_better(
                particle.fitness, particle.cv,
                particle.pbest_fitness, particle.pbest_cv,
                minimize=True
        ):
            particle.pbest_position = particle.position.copy()
            particle.pbest_fitness = particle.fitness
            particle.pbest_cv = particle.cv
            particle.pbest_is_feasible = particle.is_feasible
            return True
        return False

    def _update_global_best(self):
        """Update global best from all personal bests using Deb's rules"""
        for particle in self.swarm:
            if DebsRules.is_better(
                    particle.pbest_fitness, particle.pbest_cv,
                    self.gbest_fitness, self.gbest_cv,
                    minimize=True
            ):
                self.gbest_position = particle.pbest_position.copy()
                self.gbest_fitness = particle.pbest_fitness
                self.gbest_cv = particle.pbest_cv
                self.gbest_is_feasible = particle.pbest_cv <= 0
                self.gen_gbest_updates += 1  # Count this update

    def _compute_nfs(self) -> int:
        """
        Compute Number of Feasible Solutions based on PBEST feasibility
        NFS = count of pbests with CV <= 0
        """
        self.nfs = sum(1 for p in self.swarm if p.pbest_cv <= 0)
        return self.nfs

    def step(self):
        """
        Execute one generation of PSO with hybrid BCHM

        ALGORITHM FLOW (per paper's Algorithm 1):
        ==========================================
        G = G + 1  ← Increment generation counter at START

        For each particle i in population:
            1. Update ALL velocity components (v[i,j] for j=1 to D)
            2. Update ALL position components (x[i,j] for j=1 to D)
            3. Check if ANY component violated bounds
            4. If violated: Apply BCHM (position repair + velocity update)
            5. Evaluate particle at new position
            6. Update personal best (pbest) if current is better
            7. Update global best (gbest) if pbest is better  ← ASYNCHRONOUS UPDATE!
            8. Move to next particle

        KEY INSIGHTS:
        -------------
        1. We update ALL components of a particle before checking bounds.
           This matches the paper's vectorized equations:
           v[i] = k * (v[i] + c1*r1*(pbest[i] - x[i]) + c2*r2*(gbest - x[i]))  [ALL D components]
           x[i] = x[i] + v[i]                                                    [ALL D components]

        2. ASYNCHRONOUS gbest update: gbest is updated IMMEDIATELY after each particle.
           This means particle i+1 can use the improved gbest found by particle i
           in the SAME generation! This is different from synchronous update where
           all particles use the same gbest throughout a generation.

        ABCHS Hook Point: The BCHM (Boundary Constraint-Handling Method) is applied
        at step 4, where ABCHS subclasses override _select_method() to choose
        position_method and velocity_strategy adaptively based on current stage.
        """

        # Increment generation counter at START (more intuitive: G=1, G=2, ...)
        self.generation += 1

        # Reset generation tracking for prob_infeas metric and gbest updates
        self.gen_violated_components = 0
        self.gen_total_components = 0
        self.gen_gbest_updates = 0  # Reset gbest update counter
        self.gen_particles_improved = 0  # Reset particle improvement counter

        # ========================================================================
        # PER-PARTICLE ITERATION (Paper's Algorithm 1, lines 3-9)
        # ========================================================================
        for i, particle in enumerate(self.swarm):
            # Store old position for BCHM target reference (some methods need previous position)
            old_position = particle.position.copy()
            old_fitness = particle.fitness

            # ====================================================================
            # STEP 1: Update ALL velocity components (Paper's Equation 1)
            # ====================================================================
            # v[i,j] = k * (v[i,j] + c1*r1[j]*(pbest[i,j] - x[i,j]) + c2*r2[j]*(gbest[j] - x[i,j]))
            # where j = 1 to D (all dimensions updated simultaneously)

            r1 = np.random.rand(self.problem.dimension)  # Random vector for cognitive component
            r2 = np.random.rand(self.problem.dimension)  # Random vector for social component

            cognitive = self.c1 * r1 * (particle.pbest_position - particle.position)
            social = self.c2 * r2 * (self.gbest_position - particle.position)

            particle.velocity = self.k * (particle.velocity + cognitive + social)

            # ====================================================================
            # STEP 2: Update ALL position components (Paper's Equation 2)
            # ====================================================================
            # x[i,j] = x[i,j] + v[i,j] for j = 1 to D

            particle.position = particle.position + particle.velocity

            # ====================================================================
            # STEP 3: Check if ANY component violated bounds
            # ====================================================================
            # After updating ALL components, check which ones are out of bounds

            has_violations, violations = self.boundary_handler.check_bounds(particle.position)

            # Track boundary violations for prob_infeas metric
            # In PSO, all D dimensions are updated every generation
            self.gen_total_components += self.problem.dimension
            if has_violations:
                # Count how many dimensions violated bounds
                num_violated = np.sum(violations != 0)
                self.gen_violated_components += num_violated

            particle.was_repaired = False

            # ====================================================================
            # STEP 4: If violated, apply BCHM (Boundary Constraint-Handling Method)
            # ====================================================================
            # This is where ABCHS hooks in - subclasses override to choose method adaptively

            if has_violations:
                particle.was_repaired = True
                particle.repair_method_used = self.position_method

                # Position repair: Bring violated components back into bounds
                # Different methods: Random, Centroid, Reflection, Wrapping, etc.
                particle.position = self.boundary_handler.apply_method(
                    particle.position,
                    self.position_method,
                    target=old_position,
                    best=self.gbest_position,
                    pbest_population=self._get_pbest_population()
                )

                # Velocity update: Adjust velocity of violated components
                # Strategies: DeB (invert & damp), RaB (reinitialize), etc.
                particle.velocity = VelocityUpdater.update_velocity(
                    particle.velocity,
                    violations,
                    self.velocity_strategy
                )

            # ====================================================================
            # STEP 5: Evaluate particle at new position (Paper's Algorithm 1, line 7)
            # ====================================================================
            # f(x[i]), CV(x[i]) <- evaluate position (repaired or not)

            self._evaluate_particle(particle)

            # ====================================================================
            # STEP 6: Update personal best (Paper's Algorithm 1, line 8)
            # ====================================================================
            # Compare current position with pbest using Deb's rules:
            #   - If both feasible: compare fitness
            #   - If one feasible: feasible is better
            #   - If both infeasible: compare constraint violation

            improved = self._update_personal_best(particle)
            if improved:
                self.gen_particles_improved += 1

            # ====================================================================
            # STEP 7: Update global best IMMEDIATELY (ASYNCHRONOUS UPDATE!)
            # ====================================================================
            # This is the key difference from synchronous PSO:
            # If this particle's pbest is better than gbest, update gbest NOW
            # so that particles i+1, i+2, ... can use the new gbest in THIS generation

            if DebsRules.is_better(
                    particle.pbest_fitness, particle.pbest_cv,
                    self.gbest_fitness, self.gbest_cv,
                    minimize=True
            ):
                self.gbest_position = particle.pbest_position.copy()
                self.gbest_fitness = particle.pbest_fitness
                self.gbest_cv = particle.pbest_cv
                self.gbest_is_feasible = particle.pbest_cv <= 0
                self.gen_gbest_updates += 1  # Count this update

        # Update NFS (Number of Feasible Solutions) for ABCHS stage detection
        self._compute_nfs()

    def _get_pbest_population(self) -> List[Tuple[np.ndarray, float, float]]:
        """
        Get pbest population for methods that need it (e.g., Centroid)
        Returns list of (pbest_position, pbest_fitness, pbest_cv)
        """
        return [(p.pbest_position, p.pbest_fitness, p.pbest_cv) for p in self.swarm]

    def run(self, run_tracker=None) -> Tuple[np.ndarray, float, float]:
        """
        Run PSO until termination

        Args:
            run_tracker: Optional RunTracker for comprehensive metrics logging

        Returns:
            (best_position, best_fitness, best_cv)
        """
        while self.nfe < self.max_evaluations:
            self.step()

            # Log generation if tracker provided
            if run_tracker is not None:
                run_tracker.log_generation(self, self.problem)

        return self.gbest_position, self.gbest_fitness, self.gbest_cv

    def get_state(self) -> dict:
        """Get current state for logging/visualization"""
        return {
            'generation': self.generation,
            'nfe': self.nfe,
            'positions': np.array([p.position for p in self.swarm]),
            'velocities': np.array([p.velocity for p in self.swarm]),
            'fitness': np.array([p.fitness for p in self.swarm]),
            'cv': np.array([p.cv for p in self.swarm]),
            'pbest_positions': np.array([p.pbest_position for p in self.swarm]),
            'pbest_fitness': np.array([p.pbest_fitness for p in self.swarm]),
            'pbest_cv': np.array([p.pbest_cv for p in self.swarm]),
            'pbest_feasibility': np.array([p.pbest_cv <= 0 for p in self.swarm]),
            'gbest_position': self.gbest_position.copy(),
            'gbest_fitness': self.gbest_fitness,
            'gbest_cv': self.gbest_cv,
            'nfs': self.nfs,
            'repairs_made': sum(1 for p in self.swarm if p.was_repaired),
            'gen_violated_components': self.gen_violated_components,
            'gen_total_components': self.gen_total_components
        }
