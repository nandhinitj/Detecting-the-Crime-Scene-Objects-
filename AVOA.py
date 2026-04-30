import numpy as np
import time


#  Boundary Check
def boundary_check(X, lb, ub):
    return np.clip(X, lb, ub)


# Random Select
def random_select(v1, v2, alpha, beta):
    if np.random.rand() < alpha:
        return v1
    else:
        return v2


#  Exploration
def exploration(current, rand_vulture, F, p1, ub, lb):
    if np.random.rand() < p1:
        new_pos = rand_vulture - F * np.abs(rand_vulture - current)
    else:
        new_pos = rand_vulture + F * np.random.rand(*current.shape)

    return np.clip(new_pos, lb[0], ub[0])


#  Exploitation
def exploitation(current, best1, best2, rand_vulture, F, p1, p3, dim, ub, lb):
    if np.random.rand() < p1:
        if np.abs(F) < 0.5:
            new_pos = best1 - F * np.abs(best1 - current)
        else:
            new_pos = best2 - F * np.abs(best2 - current)
    else:
        if np.random.rand() < p3:
            new_pos = rand_vulture - F * np.abs(rand_vulture - current)
        else:
            new_pos = best1 + np.random.rand(dim) * (best2 - current)
    return np.clip(new_pos, lb[0], ub[0])


def AVOA(X, fobj, lb, ub, max_iter):
    #  African Vulture Optimization Algorithm (AVOA)
    pop_size, dim = X.shape

    Best_vulture1_X = np.zeros(dim)
    Best_vulture1_F = np.inf

    Best_vulture2_X = np.zeros(dim)
    Best_vulture2_F = np.inf

    # Parameters
    p1 = 0.6
    p2 = 0.4
    p3 = 0.6
    alpha = 0.8
    beta = 0.2
    gamma = 2.5

    convergence_curve = []

    current_iter = 0
    start_time = time.time()
    while current_iter < max_iter:
        for i in range(pop_size):
            current = X[i, :]
            fitness = fobj(current)

            if fitness < Best_vulture1_F:
                Best_vulture1_F = fitness
                Best_vulture1_X = current.copy()

            if Best_vulture1_F < fitness < Best_vulture2_F:
                Best_vulture2_F = fitness
                Best_vulture2_X = current.copy()

        #  Control Parameter
        a = np.random.uniform(-2, 2) * (
            (np.sin((np.pi / 2) * (current_iter / max_iter)) ** gamma)
            + np.cos((np.pi / 2) * (current_iter / max_iter))
            - 1
        )

        P1 = (2 * np.random.rand() + 1) * (1 - (current_iter / max_iter)) + a

        #  Update Positions
        for i in range(pop_size):
            current = X[i, :].copy()

            F = P1 * (2 * np.random.rand() - 1)

            rand_vulture = random_select(
                Best_vulture1_X, Best_vulture2_X, alpha, beta
            )

            if np.abs(F) >= 1:
                current = exploration(current, rand_vulture, F, p1, ub, lb)
            else:
                current = exploitation(
                    current,
                    Best_vulture1_X,
                    Best_vulture2_X,
                    rand_vulture,
                    F,
                    p1,
                    p3,
                    dim,
                    ub,
                    lb,
                )

            X[i, :] = current
        X = boundary_check(X, lb, ub)
        current_iter += 1
        convergence_curve.append(Best_vulture1_F)
        print(f"In Iteration {current_iter}, best estimation of the global optimum is {Best_vulture1_F:.4f}")

    total_time = time.time() - start_time
    return Best_vulture1_F, convergence_curve, Best_vulture1_X, total_time