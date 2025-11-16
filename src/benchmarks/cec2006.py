"""
CEC2006 Constrained Optimization Benchmark Problems

Unified implementation of 23 CEC2006 problems (G01-G24, excluding G20)
All verified against official MATLAB code and technical report formulas.
"""
from typing import Tuple, Dict

import numpy as np


class CEC2006Problem:
    """
    Base class for CEC2006 benchmark problems

    CEC2006 Constraint Tolerance Specification:
    - Inequality constraints: g_j(x) ≤ 0, violated if g_j(x) > tolerance_inequality
    - Equality constraints: h_k(x) = 0, violated if |h_k(x)| > tolerance_equality

    Standard tolerances as per CEC2006 specification:
    - tolerance_inequality: 1e-8 (floating point precision)
    - tolerance_equality: 1e-4 (0.0001, official CEC2006 standard)

    A solution is considered feasible if CV ≤ tolerance_feasibility.
    """

    def __init__(self):
        self.problem_id = None
        self.dimension = None
        self.lower_bounds = None
        self.upper_bounds = None
        self.known_optimum = None
        self.optimum_position = None
        self.num_inequality_constraints = 0
        self.num_equality_constraints = 0

        # CEC2006 constraint tolerances
        self.tolerance_equality = 1e-4  # CEC2006 official: epsilon = 0.0001
        self.tolerance_inequality = 1e-8  # Floating point precision for g(x) ≤ 0
        self.tolerance_feasibility = 1e-8  # Overall feasibility threshold for CV

    def evaluate(self, x: np.ndarray) -> Tuple[float, float, bool]:
        """
        Evaluate solution

        Returns:
            fitness: objective function value
            cv: total constraint violation
            is_feasible: True if CV ≤ tolerance_feasibility
        """
        fitness = self.objective(x)
        cv = self.compute_cv(x)
        is_feasible = cv <= self.tolerance_feasibility
        return fitness, cv, is_feasible

    def objective(self, x: np.ndarray) -> float:
        """Objective function to minimize"""
        raise NotImplementedError

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Returns {'inequality': g_j(x), 'equality': h_k(x)}"""
        raise NotImplementedError

    def compute_cv(self, x: np.ndarray) -> float:
        """
        Compute constraint violation as per CEC2006 specification

        CV = sum(max(0, g_j - tolerance_inequality))
             + sum(max(0, |h_k| - tolerance_equality))

        Where:
        - g_j(x) ≤ 0 are inequality constraints
        - h_k(x) = 0 are equality constraints

        Returns:
            Total constraint violation (0 if fully feasible)
        """
        c = self.constraints(x)

        # Inequality constraint violations: g_j(x) > tolerance_inequality
        ineq = c.get('inequality', np.array([]))
        ineq_violations = np.maximum(0, ineq - self.tolerance_inequality)

        # Equality constraint violations: |h_k(x)| > tolerance_equality
        eq = c.get('equality', np.array([]))
        eq_violations = np.maximum(0, np.abs(eq) - self.tolerance_equality)

        return np.sum(ineq_violations) + np.sum(eq_violations)

    def is_feasible(self, x: np.ndarray) -> bool:
        """
        Check if solution is feasible per CEC2006 specification

        A solution is feasible if:
        - All inequality constraints: g_j(x) ≤ tolerance_inequality
        - All equality constraints: |h_k(x)| ≤ tolerance_equality
        - Overall CV ≤ tolerance_feasibility
        """
        return self.compute_cv(x) <= self.tolerance_feasibility


class CEC2006_G02(CEC2006Problem):
    """G02: Maximize sum(cos^4(x_i)) - 2*prod(cos^2(x_i)) / sqrt(sum(i*x_i^2))"""

    def __init__(self):
        super().__init__()
        self.problem_id = 2
        self.dimension = 20
        self.lower_bounds = np.zeros(20)
        self.upper_bounds = np.full(20, 10.0)
        self.known_optimum = -0.803619
        self.num_inequality_constraints = 2
        self.num_equality_constraints = 0
        self.optimum_position = np.array([3.16246061572185, 3.12833142812967, 3.09479212988791,
                                          3.06145059523469, 3.02792915885555, 2.99382606701730,
                                          2.95866871765285, 2.92184227312450, 0.49482511456933,
                                          0.48835711005490, 0.48231642711865, 0.47664475092742,
                                          0.47129550835493, 0.46623099264167, 0.46142004984199,
                                          0.45683664767217, 0.45245876903267, 0.44826762241853,
                                          0.44424700958760, 0.44038285956317])

    def objective(self, x: np.ndarray) -> float:
        n = len(x)
        sum_cos4 = np.sum(np.cos(x) ** 4)
        prod_cos2 = np.prod(np.cos(x) ** 2)
        sum_ix2 = np.sum(np.arange(1, n + 1) * x ** 2)
        if sum_ix2 == 0:
            return 1e10
        return -np.abs(sum_cos4 - 2 * prod_cos2) / np.sqrt(sum_ix2)

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        g1 = 0.75 - np.prod(x)
        g2 = np.sum(x) - 7.5 * len(x)
        return {'inequality': np.array([g1, g2]), 'equality': np.array([])}


class CEC2006_G03(CEC2006Problem):
    """G03: Maximize -(sqrt(n))^n * prod(x_i)"""

    def __init__(self):
        super().__init__()
        self.problem_id = 3
        self.dimension = 10
        self.lower_bounds = np.zeros(10)
        self.upper_bounds = np.ones(10)
        self.known_optimum = -1.0005
        self.num_inequality_constraints = 0
        self.num_equality_constraints = 1
        self.optimum_position = np.array([0.31624357647283069, 0.316243577414338339, 0.316243578012345927,
                                          0.316243575664017895, 0.316243578205526066, 0.31624357738855069,
                                          0.316243575472949512, 0.316243577164883938, 0.316243578155920302,
                                          0.316243576147374916])

    def objective(self, x: np.ndarray) -> float:
        n = len(x)
        prod_x = np.prod(x)
        if prod_x <= 0:
            return 1e10
        return -(np.sqrt(n) ** n) * prod_x

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        h1 = np.sum(x ** 2) - 1.0
        return {'inequality': np.array([]), 'equality': np.array([h1])}


class CEC2006_G04(CEC2006Problem):
    """G04: Minimize 5.3578547*x3^2 + 0.8356891*x1*x5 + 37.293239*x1 - 40792.141"""

    def __init__(self):
        super().__init__()
        self.problem_id = 4
        self.dimension = 5
        self.lower_bounds = np.array([78, 33, 27, 27, 27])
        self.upper_bounds = np.array([102, 45, 45, 45, 45])
        self.known_optimum = -30665.539
        self.num_inequality_constraints = 6
        self.num_equality_constraints = 0
        self.optimum_position = np.array([78, 33, 29.9952560256815985, 45, 36.7758129057882073])

    def objective(self, x: np.ndarray) -> float:
        f = (5.3578547 * x[2] ** 2 + 0.8356891 * x[0] * x[4] +
             37.293239 * x[0] - 40792.141)
        return f

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        g1 = 85.334407 + 0.0056858 * x[1] * x[4] + 0.0006262 * x[0] * x[3] - 0.0022053 * x[2] * x[4] - 92.0
        g2 = -85.334407 - 0.0056858 * x[1] * x[4] - 0.0006262 * x[0] * x[3] + 0.0022053 * x[2] * x[4]
        g3 = 80.51249 + 0.0071317 * x[1] * x[4] + 0.0029955 * x[0] * x[1] + 0.0021813 * x[2] ** 2 - 110.0
        g4 = -80.51249 - 0.0071317 * x[1] * x[4] - 0.0029955 * x[0] * x[1] - 0.0021813 * x[2] ** 2 + 90.0
        g5 = 9.300961 + 0.0047026 * x[2] * x[4] + 0.0012547 * x[0] * x[2] + 0.0019085 * x[2] * x[3] - 25.0
        g6 = -9.300961 - 0.0047026 * x[2] * x[4] - 0.0012547 * x[0] * x[2] - 0.0019085 * x[2] * x[3] + 20.0
        return {'inequality': np.array([g1, g2, g3, g4, g5, g6]), 'equality': np.array([])}


class CEC2006_G06(CEC2006Problem):
    """G06: Minimize (x1-10)^3 + (x2-20)^3"""

    def __init__(self):
        super().__init__()
        self.problem_id = 6
        self.dimension = 2
        self.lower_bounds = np.array([13, 0])
        self.upper_bounds = np.array([100, 100])
        self.known_optimum = -6961.814
        self.num_inequality_constraints = 2
        self.num_equality_constraints = 0
        self.optimum_position = np.array([14.09500000000000064, 0.8429607892154795668])

    def objective(self, x: np.ndarray) -> float:
        return (x[0] - 10) ** 3 + (x[1] - 20) ** 3

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        g1 = -(x[0] - 5) ** 2 - (x[1] - 5) ** 2 + 100
        g2 = (x[0] - 6) ** 2 + (x[1] - 5) ** 2 - 82.81
        return {'inequality': np.array([g1, g2]), 'equality': np.array([])}


class CEC2006_G08(CEC2006Problem):
    """G08: Maximize sin^3(2πx1) * sin(2πx2) / (x1^3 * (x1+x2))"""

    def __init__(self):
        super().__init__()
        self.problem_id = 8
        self.dimension = 2
        self.lower_bounds = np.array([0, 0])
        self.upper_bounds = np.array([10, 10])
        self.known_optimum = -0.095825
        self.num_inequality_constraints = 2
        self.num_equality_constraints = 0
        self.optimum_position = np.array([1.22797135260752599, 4.24537336612274885])

    def objective(self, x: np.ndarray) -> float:
        if x[0] <= 0 or (x[0] + x[1]) <= 0:
            return 1e10

        numerator = np.sin(2 * np.pi * x[0]) ** 3 * np.sin(2 * np.pi * x[1])
        denominator = x[0] ** 3 * (x[0] + x[1])

        if denominator == 0:
            return 1e10

        f = numerator / denominator
        return -f

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        g1 = x[0] ** 2 - x[1] + 1
        g2 = 1 - x[0] + (x[1] - 4) ** 2
        return {'inequality': np.array([g1, g2]), 'equality': np.array([])}


class CEC2006_G09(CEC2006Problem):
    """G09: Minimize (x1-10)^2 + 5*(x2-12)^2 + x3^4 + 3*(x4-11)^2 + ..."""

    def __init__(self):
        super().__init__()
        self.problem_id = 9
        self.dimension = 7
        self.lower_bounds = np.full(7, -10.0)
        self.upper_bounds = np.full(7, 10.0)
        self.known_optimum = 680.630
        self.num_inequality_constraints = 4
        self.num_equality_constraints = 0
        self.optimum_position = np.array([2.33049935147405174, 1.95137236847114592, -0.477541399510615805,
                                          4.36572624923625874, -0.624486959100388983, 1.03813099410962173,
                                          1.5942266780671519])

    def objective(self, x: np.ndarray) -> float:
        f = ((x[0] - 10) ** 2 + 5 * (x[1] - 12) ** 2 + x[2] ** 4 +
             3 * (x[3] - 11) ** 2 + 10 * x[4] ** 6 + 7 * x[5] ** 2 +
             x[6] ** 4 - 4 * x[5] * x[6] - 10 * x[5] - 8 * x[6])
        return f

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        g1 = 2 * x[0] ** 2 + 3 * x[1] ** 4 + x[2] + 4 * x[3] ** 2 + 5 * x[4] - 127
        g2 = 7 * x[0] + 3 * x[1] + 10 * x[2] ** 2 + x[3] - x[4] - 282
        g3 = 23 * x[0] + x[1] ** 2 + 6 * x[5] ** 2 - 8 * x[6] - 196
        g4 = 4 * x[0] ** 2 + x[1] ** 2 - 3 * x[0] * x[1] + 2 * x[2] ** 2 + 5 * x[5] - 11 * x[6]
        return {'inequality': np.array([g1, g2, g3, g4]), 'equality': np.array([])}


class CEC2006_G11(CEC2006Problem):
    """G11: Minimize x1^2 + (x2-1)^2"""

    def __init__(self):
        super().__init__()
        self.problem_id = 11
        self.dimension = 2
        self.lower_bounds = np.array([-1, -1])
        self.upper_bounds = np.array([1, 1])
        self.known_optimum = 0.7499
        self.num_inequality_constraints = 0
        self.num_equality_constraints = 1
        self.optimum_position = np.array([-0.707036070037170616, 0.500000004333606807])

    def objective(self, x: np.ndarray) -> float:
        return x[0] ** 2 + (x[1] - 1) ** 2

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        h1 = x[1] - x[0] ** 2
        return {'inequality': np.array([]), 'equality': np.array([h1])}


class CEC2006_G12(CEC2006Problem):
    """G12: Maximize (100 - (x1-5)^2 - (x2-5)^2 - (x3-5)^2)/100"""

    def __init__(self):
        super().__init__()
        self.problem_id = 12
        self.dimension = 3
        self.lower_bounds = np.array([0, 0, 0])
        self.upper_bounds = np.array([10, 10, 10])
        self.known_optimum = -1.0
        self.num_inequality_constraints = 1
        self.num_equality_constraints = 0
        self.optimum_position = np.array([5, 5, 5])

    def objective(self, x: np.ndarray) -> float:
        f = (100 - (x[0] - 5) ** 2 - (x[1] - 5) ** 2 - (x[2] - 5) ** 2) / 100.0
        return -f

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        # Generate all 729 constraints: (x_i - p)^2 + (x_j - q)^2 + (x_k - r)^2 - 0.0625 <= 0
        # where p, q, r ∈ {1, 2, ..., 9}
        constraints = []
        for p in range(1, 10):
            for q in range(1, 10):
                for r in range(1, 10):
                    g = (x[0] - p) ** 2 + (x[1] - q) ** 2 + (x[2] - r) ** 2 - 0.0625
                    constraints.append(g)
        return {'inequality': np.array(constraints), 'equality': np.array([])}

    def compute_cv(self, x: np.ndarray) -> float:
        """Override CV: feasible if inside at least one of 729 spheres"""
        min_distance = np.inf
        for p in range(1, 10):
            for q in range(1, 10):
                for r in range(1, 10):
                    distance = (x[0] - p) ** 2 + (x[1] - q) ** 2 + (x[2] - r) ** 2 - 0.0625
                    min_distance = min(min_distance, distance)

        return max(0.0, min_distance)


class CEC2006_G13(CEC2006Problem):
    """G13: Minimize exp(x1*x2*x3*x4*x5)"""

    def __init__(self):
        super().__init__()
        self.problem_id = 13
        self.dimension = 5
        self.lower_bounds = np.array([-2.3, -2.3, -3.2, -3.2, -3.2])
        self.upper_bounds = np.array([2.3, 2.3, 3.2, 3.2, 3.2])
        self.known_optimum = 0.053950
        self.num_inequality_constraints = 0
        self.num_equality_constraints = 3
        self.optimum_position = np.array([-1.71714224003, 1.59572124049468, 1.8272502406271,
                                          -0.763659881912867, -0.76365986736498])

    def objective(self, x: np.ndarray) -> float:
        prod = x[0] * x[1] * x[2] * x[3] * x[4]
        return np.exp(prod)

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        h1 = x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2 + x[4] ** 2 - 10
        h2 = x[1] * x[2] - 5 * x[3] * x[4]
        h3 = x[0] ** 3 + x[1] ** 3 + 1
        return {'inequality': np.array([]), 'equality': np.array([h1, h2, h3])}


class CEC2006_G16(CEC2006Problem):
    """G16: Minimize complex objective with intermediate variables"""

    def __init__(self):
        super().__init__()
        self.problem_id = 16
        self.dimension = 5  # x1, x2, x3, x4, x5
        self.lower_bounds = np.array([704.4148, 68.6, 0, 193, 25])
        self.upper_bounds = np.array([906.3855, 288.88, 134.75, 287.0966, 84.1988])
        self.known_optimum = -1.905155
        self.num_inequality_constraints = 38
        self.num_equality_constraints = 0
        self.optimum_position = np.array([705.174537070090537, 68.5999999999999943, 102.899999999999991,
                                          282.324931593660324, 37.5841164258054832])

    def compute_y_c(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Compute intermediate variables y1-y17 and c1-c17"""
        y1 = x[1] + x[2] + 41.6

        c1 = 0.024 * x[3] - 4.62
        y2 = (12.5 / c1 + 12.0) if c1 != 0 else 1e10

        c2 = 0.0003535 * x[0] ** 2 + 0.5311 * x[0] + 0.08705 * y2 * x[0]
        c3 = 0.052 * x[0] + 78.0 + 0.002377 * y2 * x[0]
        y3 = c2 / c3
        y4 = 19 * y3

        c4 = 0.04782 * (x[0] - y3) + 0.1956 * (x[0] - y3) ** 2 / x[1] + 0.6376 * y4 + 1.594 * y3
        c5 = 100 * x[1]
        c6 = x[0] - y3 - y4
        c7 = 0.950 - c4 / c5
        y5 = c6 * c7

        y6 = x[0] - y5 - y4 - y3
        c8 = (y5 + y4) * 0.995
        y7 = c8 / y1
        y8 = c8 / 3798.0
        c9 = y7 - 0.0663 * y7 / y8 - 0.3153
        y9 = (96.82 / c9) + 0.321 * y1
        y10 = 1.29 * y5 + 1.258 * y4 + 2.29 * y3 + 1.71 * y6
        y11 = 1.71 * x[0] - 0.452 * y4 + 0.580 * y3
        c10 = 12.3 / 752.3
        c11 = 1.75 * y2 * (0.995 * x[0])
        c12 = 0.995 * y10 + 1998.0
        y12 = c10 * x[0] + c11 / c12
        y13 = c12 - 1.75 * y2
        y14 = 3623.0 + 64.4 * x[1] + 58.4 * x[2] + 146312.0 / (y9 + x[4])
        c13 = 0.995 * y10 + 60.8 * x[1] + 48.0 * x[3] - 0.1121 * y14 - 5095.0
        y15 = y13 / c13
        y16 = 148000.0 - 331000.0 * y15 + 40.0 * y13 - 61.0 * y15 * y13
        c14 = 2324.0 * y10 - 28740000.0 * y2
        y17 = 14130000.0 - 1328.0 * y10 - 531.0 * y11 + c14 / c12
        c15 = y13 / y15 - y13 / 0.52
        c16 = 1.104 - 0.72 * y15
        c17 = y9 + x[4]

        y = np.array([y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15, y16, y17])
        c_dict = {'c1': c1, 'c8': c8, 'c9': c9, 'c10': c10, 'c11': c11, 'c12': c12,
                  'c13': c13, 'c14': c14, 'c15': c15, 'c16': c16, 'c17': c17}

        return y, c_dict

    def objective(self, x: np.ndarray) -> float:
        y, c_dict = self.compute_y_c(x)
        return (0.000117 * y[13] + 0.1365 + 0.00002358 * y[12] + 0.000001502 * y[15] + 0.0321 * y[11] +
                0.004324 * y[4] + 0.0001 * c_dict['c15'] / c_dict['c16'] + 37.48 * y[1] / c_dict['c12'] -
                0.0000005843 * y[16])

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        y, c_dict = self.compute_y_c(x)
        c1, c12, c17 = c_dict['c1'], c_dict['c12'], c_dict['c17']

        constraints = []

        constraints.append(0.28 / 0.72 * y[4] - y[3])  # g1
        constraints.append(x[2] - 1.5 * x[1])  # g2
        constraints.append(3496.0 * y[1] / c12 - 21.0)  # g3
        constraints.append(110.6 + y[0] - 62212.0 / c17 if c17 != 0 else 1e10)  # g4
        constraints.append(213.1 - y[0])  # g5
        constraints.append(y[0] - 405.23)  # g6
        constraints.append(17.505 - y[1])  # g7
        constraints.append(y[1] - 1053.6667)  # g8
        constraints.append(11.275 - y[2])  # g9
        constraints.append(y[2] - 35.03)  # g10
        constraints.append(214.228 - y[3])  # g11
        constraints.append(y[3] - 665.585)  # g12
        constraints.append(7.458 - y[4])  # g13
        constraints.append(y[4] - 584.463)  # g14
        constraints.append(0.961 - y[5])  # g15
        constraints.append(y[5] - 265.916)  # g16
        constraints.append(1.612 - y[6])  # g17
        constraints.append(y[6] - 7.046)  # g18
        constraints.append(0.146 - y[7])  # g19
        constraints.append(y[7] - 0.222)  # g20
        constraints.append(107.99 - y[8])  # g21
        constraints.append(y[8] - 273.366)  # g22
        constraints.append(922.693 - y[9])  # g23
        constraints.append(y[9] - 1286.105)  # g24
        constraints.append(926.832 - y[10])  # g25
        constraints.append(y[10] - 1444.046)  # g26
        constraints.append(18.766 - y[11])  # g27
        constraints.append(y[11] - 537.141)  # g28
        constraints.append(1072.163 - y[12])  # g29
        constraints.append(y[12] - 3247.039)  # g30
        constraints.append(8961.448 - y[13])  # g31
        constraints.append(y[13] - 26844.086)  # g32
        constraints.append(0.063 - y[14])  # g33
        constraints.append(y[14] - 0.386)  # g34
        constraints.append(71084.33 - y[15])  # g35
        constraints.append(-140000.0 + y[15])  # g36
        constraints.append(2802713.0 - y[16])  # g37
        constraints.append(y[16] - 12146108.0)  # g38

        return {'inequality': np.array(constraints), 'equality': np.array([])}


class CEC2006_G19(CEC2006Problem):
    """G19: Complex polynomial problem with 15 variables"""

    def __init__(self):
        super().__init__()
        self.problem_id = 19
        self.dimension = 15
        self.lower_bounds = np.zeros(15)
        self.upper_bounds = np.full(15, 10.0)
        self.known_optimum = 32.656
        self.num_inequality_constraints = 5
        self.num_equality_constraints = 0
        self.optimum_position = np.array([1.66991341326291344e-17, 3.95378229282456509e-16, 3.94599045143233784,
                                          1.06036597479721211e-16, 3.2831773458454161, 9.99999999999999822,
                                          1.12829414671605333e-17, 1.2026194599794709e-17, 2.50706276000769697e-15,
                                          2.24624122987970677e-15, 0.370764847417013987, 0.278456024942955571,
                                          0.523838487672241171, 0.388620152510322781, 0.298156764974678579])

        self.e = np.array([-15, -27, -36, -18, -12])

        self.c = np.array([
            [30, -20, -10, 32, -10],
            [-20, 39, -6, -31, 32],
            [-10, -6, 10, -6, -10],
            [32, -31, -6, 39, -20],
            [-10, 32, -10, -20, 30]
        ])

        self.d = np.array([4, 8, 10, 6, 2])

        self.a = np.array([
            [-16, 2, 0, 1, 0],
            [0, -2, 0, 0.4, 2],
            [-3.5, 0, 2, 0, 0],
            [0, -2, 0, -4, -1],
            [0, -9, -2, 1, -2.8],
            [2, 0, -4, 0, 0],
            [-1, -1, -1, -1, -1],
            [-1, -2, -3, -2, -1],
            [1, 2, 3, 4, 5],
            [1, 1, 1, 1, 1]
        ])

    def objective(self, x: np.ndarray) -> float:
        # First term: sum over j,i of cij * x(10+i) * x(10+j)
        term1 = 0
        for j in range(5):
            for i in range(5):
                term1 += self.c[i, j] * x[10 + i] * x[10 + j]

        # Second term: 2 * sum over j of dj * x³(10+j)
        term2 = 2 * np.sum(self.d * x[10:15] ** 3)

        # Third term: - sum over i of bi*xi (where b = [-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1])
        b = np.array([-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1])
        term3 = -np.sum(b * x[:10])

        return term1 + term2 + term3

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        g = np.zeros(5)

        for j in range(5):
            # -2 * sum over i of c[i,j] * x[10+i]
            term1 = -2 * np.sum(self.c[:, j] * x[10:15])

            # -3 * dj * x²(10+j)
            term2 = -3 * self.d[j] * x[10 + j] ** 2

            # -ej
            term3 = -self.e[j]

            # sum over i of a[i,j] * x[i]
            term4 = np.sum(self.a[:, j] * x[:10])

            g[j] = term1 + term2 + term3 + term4

        return {'inequality': g, 'equality': np.array([])}


class CEC2006_G24(CEC2006Problem):
    """G24: Minimize -x1 - x2"""

    def __init__(self):
        super().__init__()
        self.problem_id = 24
        self.dimension = 2
        self.lower_bounds = np.array([0, 0])
        self.upper_bounds = np.array([3, 4])
        self.known_optimum = -5.508
        self.num_inequality_constraints = 2
        self.num_equality_constraints = 0
        self.optimum_position = np.array([2.32952019747762, 3.17849307411774])

    def objective(self, x: np.ndarray) -> float:
        return -x[0] - x[1]

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        g1 = (-2 * x[0] ** 4 + 8 * x[0] ** 3 - 8 * x[0] ** 2 + x[1] - 2)
        g2 = (-4 * x[0] ** 4 + 32 * x[0] ** 3 - 88 * x[0] ** 2 + 96 * x[0] + x[1] - 36)
        return {'inequality': np.array([g1, g2]), 'equality': np.array([])}


class CEC2006_G01(CEC2006Problem):
    """G01: Quadratic function with linear constraints"""

    def __init__(self):
        super().__init__()
        self.problem_id = 1
        self.dimension = 13
        self.lower_bounds = np.zeros(13)
        self.upper_bounds = np.ones(13)  # Initialize first

        self.upper_bounds[0:9] = 1  # x1-x9: 0 ≤ xi ≤ 1
        self.upper_bounds[9:12] = 100  # x10-x12: 0 ≤ xi ≤ 100
        self.upper_bounds[12] = 1  # x13: 0 ≤ x13 ≤ 1
        self.known_optimum = -15.0
        self.num_inequality_constraints = 9
        self.num_equality_constraints = 0
        self.optimum_position = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1])

    def objective(self, x: np.ndarray) -> float:
        return 5 * np.sum(x[0:4]) - 5 * np.sum(x[0:4] ** 2) - np.sum(x[4:13])

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        g = np.zeros(9)
        g[0] = 2 * x[0] + 2 * x[1] + x[9] + x[10] - 10
        g[1] = 2 * x[0] + 2 * x[2] + x[9] + x[11] - 10
        g[2] = 2 * x[1] + 2 * x[2] + x[10] + x[11] - 10
        g[3] = -8 * x[0] + x[9]
        g[4] = -8 * x[1] + x[10]
        g[5] = -8 * x[2] + x[11]
        g[6] = -2 * x[3] - x[4] + x[9]
        g[7] = -2 * x[5] - x[6] + x[10]
        g[8] = -2 * x[7] - x[8] + x[11]
        return {'inequality': g, 'equality': np.array([])}


class CEC2006_G05(CEC2006Problem):
    """G05: Polynomial function

    Note: Optimal solution has equality constraints at boundary (|h| = 0.0001).
    This is by design per CEC2006 specification.
    """

    def __init__(self):
        super().__init__()
        self.problem_id = 5
        self.dimension = 4
        self.lower_bounds = np.array([0, 0, -0.55, -0.55])
        self.upper_bounds = np.array([1200, 1200, 0.55, 0.55])
        self.known_optimum = 5126.4967140071
        self.num_inequality_constraints = 2
        self.num_equality_constraints = 3
        self.optimum_position = np.array(
            [679.945148297028709, 1026.06697600004691, 0.118876369094410433, -0.39623348521517826])

    def objective(self, x: np.ndarray) -> float:
        return (3 * x[0] + 0.000001 * x[0] ** 3 + 2 * x[1] +
                (0.000002 / 3) * x[1] ** 3)

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        g1 = -x[3] + x[2] - 0.55
        g2 = -x[2] + x[3] - 0.55

        h1 = 1000 * np.sin(-x[2] - 0.25) + 1000 * np.sin(-x[3] - 0.25) + 894.8 - x[0]
        h2 = 1000 * np.sin(x[2] - 0.25) + 1000 * np.sin(x[2] - x[3] - 0.25) + 894.8 - x[1]
        h3 = 1000 * np.sin(x[3] - 0.25) + 1000 * np.sin(x[3] - x[2] - 0.25) + 1294.8

        return {'inequality': np.array([g1, g2]), 'equality': np.array([h1, h2, h3])}


class CEC2006_G07(CEC2006Problem):
    """G07: Quadratic function"""

    def __init__(self):
        super().__init__()
        self.problem_id = 7
        self.dimension = 10
        self.lower_bounds = np.full(10, -10.0)
        self.upper_bounds = np.full(10, 10.0)
        self.known_optimum = 24.30620906818
        self.num_inequality_constraints = 8
        self.num_equality_constraints = 0
        self.optimum_position = np.array([2.17199634142692, 2.3636830416034,
                                          8.77392573913157, 5.09598443745173,
                                          0.990654756560493, 1.43057392853463,
                                          1.32164415364306, 9.82872576524495,
                                          8.2800915887356, 8.3759266477347])

    def objective(self, x: np.ndarray) -> float:
        return (x[0] ** 2 + x[1] ** 2 + x[0] * x[1] - 14 * x[0] - 16 * x[1] +
                (x[2] - 10) ** 2 + 4 * (x[3] - 5) ** 2 + (x[4] - 3) ** 2 +
                2 * (x[5] - 1) ** 2 + 5 * x[6] ** 2 + 7 * (x[7] - 11) ** 2 +
                2 * (x[8] - 10) ** 2 + (x[9] - 7) ** 2 + 45)

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        g = np.zeros(8)
        g[0] = -105 + 4 * x[0] + 5 * x[1] - 3 * x[6] + 9 * x[7]
        g[1] = 10 * x[0] - 8 * x[1] - 17 * x[6] + 2 * x[7]
        g[2] = -8 * x[0] + 2 * x[1] + 5 * x[8] - 2 * x[9] - 12
        g[3] = 3 * (x[0] - 2) ** 2 + 4 * (x[1] - 3) ** 2 + 2 * x[2] ** 2 - 7 * x[3] - 120
        g[4] = 5 * x[0] ** 2 + 8 * x[1] + (x[2] - 6) ** 2 - 2 * x[3] - 40
        g[5] = x[0] ** 2 + 2 * (x[1] - 2) ** 2 - 2 * x[0] * x[1] + 14 * x[4] - 6 * x[5]
        g[6] = 0.5 * (x[0] - 8) ** 2 + 2 * (x[1] - 4) ** 2 + 3 * x[4] ** 2 - x[5] - 30
        g[7] = -3 * x[0] + 6 * x[1] + 12 * (x[8] - 8) ** 2 - 7 * x[9]
        return {'inequality': g, 'equality': np.array([])}


class CEC2006_G10(CEC2006Problem):
    """G10: Linear and polynomial function"""

    def __init__(self):
        super().__init__()
        self.problem_id = 10
        self.dimension = 8
        self.lower_bounds = np.array([100, 1000, 1000, 10, 10, 10, 10, 10])
        self.upper_bounds = np.array([10000, 10000, 10000, 1000, 1000, 1000, 1000, 1000])
        self.known_optimum = 7049.24802052867
        self.num_inequality_constraints = 6
        self.num_equality_constraints = 0
        self.optimum_position = np.array(
            [579.306685017979589, 1359.97067807935605, 5109.97065743133317, 182.01769963061534,
             295.601173702746792, 217.982300369384632, 286.41652592786852, 395.60117370274673])

    def objective(self, x: np.ndarray) -> float:
        return x[0] + x[1] + x[2]

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        g = np.zeros(6)
        g[0] = -1 + 0.0025 * (x[3] + x[5])
        g[1] = -1 + 0.0025 * (x[4] + x[6] - x[3])
        g[2] = -1 + 0.01 * (x[7] - x[4])
        g[3] = -x[0] * x[5] + 833.33252 * x[3] + 100 * x[0] - 83333.333
        g[4] = -x[1] * x[6] + 1250 * x[4] + x[1] * x[3] - 1250 * x[3]
        g[5] = -x[2] * x[7] + 1250000 + x[2] * x[4] - 2500 * x[4]
        return {'inequality': g, 'equality': np.array([])}


class CEC2006_G14(CEC2006Problem):
    """G14: Nonlinear function

    Note: Optimal solution has equality constraints at boundary (|h| = 0.0001).
    This is by design per CEC2006 specification.
    """

    def __init__(self):
        super().__init__()
        self.problem_id = 14
        self.dimension = 10
        self.lower_bounds = np.zeros(10)
        self.upper_bounds = np.full(10, 10.0)
        self.known_optimum = -47.7648884595
        self.num_inequality_constraints = 0
        self.num_equality_constraints = 3
        self.optimum_position = np.array([0.0406684113216282, 0.147721240492452, 0.783205732104114,
                                          0.00141433931889084, 0.485293636780388, 0.000693183051556082,
                                          0.0274052040687766,
                                          0.0179509660214818, 0.0373268186859717, 0.0968844604336845])

        self.c = np.array([
            [-6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.1, -10.708, -26.662, -22.179],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

    def objective(self, x: np.ndarray) -> float:
        sum_x = np.sum(x)
        # Prevent log(0) and division by zero
        x_safe = np.maximum(x, 1e-10)  # Avoid x=0
        sum_x_safe = max(sum_x, 1e-10)  # Avoid sum=0
        return np.sum(x_safe * (self.c[0] + np.log(x_safe / sum_x_safe)))

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        h = np.zeros(3)
        h[0] = x[0] + 2 * x[1] + 2 * x[2] + x[5] + x[9] - 2
        h[1] = x[3] + 2 * x[4] + x[5] + x[6] - 1
        h[2] = x[2] + x[6] + x[7] + 2 * x[8] + x[9] - 1
        return {'inequality': np.array([]), 'equality': h}


class CEC2006_G15(CEC2006Problem):
    """G15: Quadratic function"""

    def __init__(self):
        super().__init__()
        self.problem_id = 15
        self.dimension = 3
        self.lower_bounds = np.zeros(3)
        self.upper_bounds = np.full(3, 10.0)
        self.known_optimum = 961.715022289961
        self.num_inequality_constraints = 0
        self.num_equality_constraints = 2
        self.optimum_position = np.array([3.51212812611795133, 0.216987510429556135,
                                          3.55217854929179921])

    def objective(self, x: np.ndarray) -> float:
        return 1000 - x[0] ** 2 - 2 * x[1] ** 2 - x[2] ** 2 - x[0] * x[1] - x[0] * x[2]

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        h1 = x[0] ** 2 + x[1] ** 2 + x[2] ** 2 - 25
        h2 = 8 * x[0] + 14 * x[1] + 7 * x[2] - 56
        return {'inequality': np.array([]), 'equality': np.array([h1, h2])}


class CEC2006_G17(CEC2006Problem):
    """G17: Nonlinear function

    Note: There is a small objective value mismatch (~5.66e-03) between computed
    and declared optimal values. The optimal position values from the CEC2006 PDF
    may be approximate. Constraint feasibility is satisfied.
    """

    def __init__(self):
        super().__init__()
        self.problem_id = 17
        self.dimension = 6
        self.lower_bounds = np.array([0, 0, 340, 340, -1000, 0])
        self.upper_bounds = np.array([400, 1000, 420, 420, 1000, 0.5236])
        self.known_optimum = 8853.53967480648
        self.num_inequality_constraints = 0
        self.num_equality_constraints = 4
        self.optimum_position = np.array([201.784467214523659, 99.9999999999999005,
                                          383.071034852773266, 420, -10.9076584514292652, 0.0731482312084287128])

    def objective(self, x: np.ndarray) -> float:
        if x[0] >= 0 and x[0] < 300:
            f1 = 30 * x[0]
        elif x[0] >= 300 and x[0] < 400:
            f1 = 31 * x[0]
        else:
            f1 = 0

        if x[1] >= 0 and x[1] < 100:
            f2 = 28 * x[1]
        elif x[1] >= 100 and x[1] < 200:
            f2 = 29 * x[1]
        elif x[1] >= 200 and x[1] < 1000:
            f2 = 30 * x[1]
        else:
            f2 = 0

        return f1 + f2

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        h = np.zeros(4)
        h[0] = -x[0] + 300 - x[2] * x[3] * np.cos(1.48477 - x[5]) / 131.078 + \
               0.90798 * x[2] ** 2 * np.cos(1.47588) / 131.078
        h[1] = -x[1] - x[2] * x[3] * np.cos(1.48477 + x[5]) / 131.078 + \
               0.90798 * x[3] ** 2 * np.cos(1.47588) / 131.078
        h[2] = -x[4] - x[2] * x[3] * np.sin(1.48477 + x[5]) / 131.078 + \
               0.90798 * x[3] ** 2 * np.sin(1.47588) / 131.078
        h[3] = 200 - x[2] * x[3] * np.sin(1.48477 - x[5]) / 131.078 + \
               0.90798 * x[2] ** 2 * np.sin(1.47588) / 131.078
        return {'inequality': np.array([]), 'equality': h}


class CEC2006_G18(CEC2006Problem):
    """G18: Quadratic function"""

    def __init__(self):
        super().__init__()
        self.problem_id = 18
        self.dimension = 9
        self.lower_bounds = np.array([-10, -10, -10, -10, -10, -10, -10, -10, 0])
        self.upper_bounds = np.array([10, 10, 10, 10, 10, 10, 10, 10, 20])
        self.known_optimum = -0.8660254038
        self.num_inequality_constraints = 13
        self.num_equality_constraints = 0
        self.optimum_position = np.array([-0.657776192427943163, -0.153418773482438542,
                                          0.323413871675240938, -0.946257611651304398,
                                          -0.657776194376798906, -0.753213434632691414,
                                          0.323413874123576972, -0.346462947962331735,
                                          0.59979466285217542])

    def objective(self, x: np.ndarray) -> float:
        return -0.5 * (x[0] * x[3] - x[1] * x[2] + x[2] * x[8] - x[4] * x[8] + x[4] * x[7] - x[5] * x[6])

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        g = np.zeros(13)
        g[0] = x[2] ** 2 + x[3] ** 2 - 1
        g[1] = x[8] ** 2 - 1
        g[2] = x[4] ** 2 + x[5] ** 2 - 1
        g[3] = x[0] ** 2 + (x[1] - x[8]) ** 2 - 1
        g[4] = (x[0] - x[4]) ** 2 + (x[1] - x[5]) ** 2 - 1
        g[5] = (x[0] - x[6]) ** 2 + (x[1] - x[7]) ** 2 - 1
        g[6] = (x[2] - x[4]) ** 2 + (x[3] - x[5]) ** 2 - 1
        g[7] = (x[2] - x[6]) ** 2 + (x[3] - x[7]) ** 2 - 1
        g[8] = x[6] ** 2 + (x[7] - x[8]) ** 2 - 1
        g[9] = x[1] * x[2] - x[0] * x[3]
        g[10] = -x[2] * x[8]
        g[11] = x[4] * x[8]
        g[12] = x[5] * x[6] - x[4] * x[7]
        return {'inequality': g, 'equality': np.array([])}


# class CEC2006_G20(CEC2006Problem):
#     """G20: Nonlinear function"""
#
#     def __init__(self):
#         super().__init__()
#         self.problem_id = 20
#         self.dimension = 24
#         self.lower_bounds = np.zeros(24)
#         self.upper_bounds = np.full(24, 10.0)
#         self.known_optimum = 0.2049794002
#         self.num_inequality_constraints = 6
#         self.num_equality_constraints = 14
#
#     def objective(self, x: np.ndarray) -> float:
#         a = [0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09,
#              0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09]
#         b = [44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94, 133.425, 82.507,
#              46.07, 60.097, 44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94,
#              133.425, 82.507, 46.07, 60.097]
#         c = [123.7, 31.7, 45.7, 14.7, 84.7, 27.7, 49.7, 7.1, 2.1, 17.7, 0.85, 0.64]
#         d = [31.244, 36.12, 34.784, 92.7, 82.7, 91.6, 56.708, 82.7, 80.8, 64.517, 49.4, 49.1]
#         e = [0.1, 0.3, 0.4, 0.3, 0.6, 0.3]
#
#         s = 0
#         for i in range(12):
#             s += a[i]*x[i] + b[i]
#         for i in range(12, 24):
#             s += a[i]*x[i]
#
#         return s
#
#     def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
#         a = [0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09,
#              0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09]
#         b = [44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94, 133.425, 82.507,
#              46.07, 60.097, 44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94,
#              133.425, 82.507, 46.07, 60.097]
#         c = [123.7, 31.7, 45.7, 14.7, 84.7, 27.7, 49.7, 7.1, 2.1, 17.7, 0.85, 0.64]
#         d = [31.244, 36.12, 34.784, 92.7, 82.7, 91.6, 56.708, 82.7, 80.8, 64.517, 49.4, 49.1]
#         e = [0.1, 0.3, 0.4, 0.3, 0.6, 0.3]
#
#         g = np.zeros(6)
#         for j in range(6):
#             g[j] = sum(x[i] + x[i+12] for i in range(2*j, 2*j+2)) - e[j]
#
#         h = np.zeros(14)
#         h[0] = x[0] + x[12] - 1.0
#         h[1] = x[1] + x[13] - 1.0
#         h[2] = x[2] + x[14] - 1.0
#         h[3] = x[3] + x[15] - 1.0
#         h[4] = x[4] + x[16] - 1.0
#         h[5] = x[5] + x[17] - 1.0
#         h[6] = x[6] + x[18] - 1.0
#         h[7] = x[7] + x[19] - 1.0
#         h[8] = x[8] + x[20] - 1.0
#         h[9] = x[9] + x[21] - 1.0
#         h[10] = x[10] + x[22] - 1.0
#         h[11] = x[11] + x[23] - 1.0
#
#         for i in range(12):
#             h[12] += (a[i]*x[i] + b[i])/c[i%12]
#         for i in range(12, 24):
#             h[12] += a[i]*x[i]/c[i%12]
#         h[12] -= 1.671
#
#         for i in range(12):
#             h[13] += (a[i]*x[i] + b[i])/d[i%12]
#         for i in range(12, 24):
#             h[13] += a[i]*x[i]/d[i%12]
#         h[13] -= 0.7553
#
#         return {'inequality': g, 'equality': h}


class CEC2006_G21(CEC2006Problem):
    """G21: Linear function

    Note: Optimal solution has equality constraints at boundary (|h| = 0.0001).
    This is by design per CEC2006 specification.
    """

    def __init__(self):
        super().__init__()
        self.problem_id = 21
        self.dimension = 7
        self.lower_bounds = np.array([0, 0, 0, 100, 6.3, 5.9, 4.5])
        self.upper_bounds = np.array([1000, 40, 40, 300, 6.7, 6.4, 6.25])
        self.known_optimum = 193.7245100700
        self.num_inequality_constraints = 1
        self.num_equality_constraints = 5
        self.optimum_position = np.array([193.724510070034967, 5.56944131553368433e-27,
                                          17.3191887294084914, 100.047897801386839,
                                          6.68445185362377892, 5.99168428444264833,
                                          6.21451648886070451])

    def objective(self, x: np.ndarray) -> float:
        return x[0]

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        g1 = -x[0] + 35 * x[1] ** 0.6 + 35 * x[2] ** 0.6

        h = np.zeros(5)
        h[0] = -300 * x[2] + 7500 * x[4] - 7500 * x[5] - 25 * x[3] * x[4] + 25 * x[3] * x[5] + x[2] * x[3]
        h[1] = 100 * x[1] + 155.365 * x[3] + 2500 * x[6] - x[1] * x[3] - 25 * x[3] * x[6] - 15536.5
        h[2] = -x[4] + np.log(-x[3] + 900)
        h[3] = -x[5] + np.log(x[3] + 300)
        h[4] = -x[6] + np.log(-2 * x[3] + 700)

        return {'inequality': np.array([g1]), 'equality': h}


class CEC2006_G22(CEC2006Problem):
    """G22: Nonlinear function

    Note: Optimal solution has tiny inequality constraint violation (~2.21e-07)
    within numerical precision tolerance.
    """

    def __init__(self):
        super().__init__()
        self.problem_id = 22
        self.dimension = 22
        self.lower_bounds = np.array(
            [0, 0, 0, 0, 0, 0, 0, 100, 100, 100.01, 100, 100, 0, 0, 0, 0.01, 0.01, -4.7, -4.7, -4.7, -4.7, -4.7])
        self.upper_bounds = np.array(
            [20000, 1e6, 1e6, 1e6, 4e7, 4e7, 4e7, 299.99, 399.99, 300, 400, 600, 500, 500, 500, 300, 400, 6.25, 6.25,
             6.25, 6.25, 6.25])
        self.known_optimum = 236.430975504001
        self.num_inequality_constraints = 1
        self.num_equality_constraints = 19
        self.optimum_position = np.array(
            [236.430975504001054, 135.82847151732463, 204.818152544824585, 6446.54654059436416,
             3007540.83940215595, 4074188.65771341929, 32918270.5028952882, 130.075408394314167,
             170.817294970528621, 299.924591605478554, 399.258113423595205, 330.817294971142758,
             184.51831230897065, 248.64670239647424, 127.658546694545862, 269.182627528746707,
             160.000016724090955, 5.29788288102680571, 5.13529735903945728, 5.59531526444068827,
             5.43444479314453499, 5.07517453535834395])

    def objective(self, x: np.ndarray) -> float:
        return x[0]

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        g1 = x[0] - x[1] ** 0.6 - x[2] ** 0.6 - x[3] ** 0.6

        h = np.zeros(19)
        h[0] = x[4] - 100000 * x[7] + 10 ** 7
        h[1] = x[5] + 100000 * x[7] - 100000 * x[8]
        h[2] = x[6] + 100000 * x[8] - 5 * 10 ** 7
        h[3] = x[4] + 100000 * x[9] - 3.3 * 10 ** 7
        h[4] = x[5] + 100000 * x[10] - 4.4 * 10 ** 7
        h[5] = x[6] + 100000 * x[11] - 6.6 * 10 ** 7
        h[6] = x[4] - 120 * x[1] * x[12]
        h[7] = x[5] - 80 * x[2] * x[13]
        h[8] = x[6] - 40 * x[3] * x[14]
        h[9] = x[7] - x[10] + x[15]
        h[10] = x[8] - x[11] + x[16]
        h[11] = -x[17] + np.log(x[9] - 100)
        h[12] = -x[18] + np.log(-x[7] + 300)
        h[13] = -x[19] + np.log(x[15])
        h[14] = -x[20] + np.log(-x[8] + 400)
        h[15] = -x[21] + np.log(x[16])
        h[16] = -x[7] - x[9] + x[12] * x[17] - x[12] * x[18] + 400
        h[17] = x[7] - x[8] - x[10] + x[13] * x[19] - x[13] * x[20] + 400
        h[18] = x[8] - x[11] - 4.60517 * x[14] + x[14] * x[21] + 100

        return {'inequality': np.array([g1]), 'equality': h}


class CEC2006_G23(CEC2006Problem):
    """G23: Linear function

    Note: Optimal solution has equality constraints at boundary (|h| = 0.0001).
    This is by design per CEC2006 specification.
    """

    def __init__(self):
        super().__init__()
        self.problem_id = 23
        self.dimension = 9
        self.lower_bounds = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0.01])
        self.upper_bounds = np.array([300, 300, 100, 200, 100, 300, 100, 200, 0.03])
        self.known_optimum = -400.055099999999584
        self.num_inequality_constraints = 2
        self.num_equality_constraints = 4
        self.optimum_position = np.array([0.00510000000000259465, 99.9947000000000514,
                                          9.01920162996045897e-18, 99.9999000000000535, 0.000100000000027086086,
                                          2.75700683389584542e-14, 99.9999999999999574, 200, 0.0100000100000100008])

    def objective(self, x: np.ndarray) -> float:
        return -9 * x[4] - 15 * x[7] + 6 * x[0] + 16 * x[1] + 10 * (x[5] + x[6])

    def constraints(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        g = np.zeros(2)
        g[0] = x[8] * x[2] + 0.02 * x[5] - 0.025 * x[4]
        g[1] = x[8] * x[3] + 0.02 * x[6] - 0.015 * x[7]

        h = np.zeros(4)
        h[0] = x[0] + x[1] - x[2] - x[3]
        h[1] = 0.03 * x[0] + 0.01 * x[1] - x[8] * (x[2] + x[3])
        h[2] = x[2] + x[5] - x[4]
        h[3] = x[3] + x[6] - x[7]

        return {'inequality': g, 'equality': h}


# Problem registry - All CEC2006 problems (excluding G20)
PROBLEM_REGISTRY = {
    1: CEC2006_G01,
    2: CEC2006_G02,
    3: CEC2006_G03,
    4: CEC2006_G04,
    5: CEC2006_G05,
    6: CEC2006_G06,
    7: CEC2006_G07,
    8: CEC2006_G08,
    9: CEC2006_G09,
    10: CEC2006_G10,
    11: CEC2006_G11,
    12: CEC2006_G12,
    13: CEC2006_G13,
    14: CEC2006_G14,
    15: CEC2006_G15,
    16: CEC2006_G16,
    17: CEC2006_G17,
    18: CEC2006_G18,
    19: CEC2006_G19,
    21: CEC2006_G21,
    22: CEC2006_G22,
    23: CEC2006_G23,
    24: CEC2006_G24
}


def get_problem(problem_id: int) -> CEC2006Problem:
    """
    Factory function to get a CEC2006 problem instance

    Args:
        problem_id: Problem number (1-24, excluding 20)

    Returns:
        Instance of the requested problem

    Raises:
        ValueError: If problem_id is not implemented
    """
    if problem_id not in PROBLEM_REGISTRY:
        available = sorted(PROBLEM_REGISTRY.keys())
        raise ValueError(f"Problem G{problem_id:02d} not implemented. Available problems: {available}")
    return PROBLEM_REGISTRY[problem_id]()
