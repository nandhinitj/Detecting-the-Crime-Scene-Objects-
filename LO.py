import numpy as np
import time


def LO(initsol, objfun, xmin, xmax, Max_iter):
    # Lemurs Optimizer (LO)
    # Parameters
    Npop, dim = initsol.shape
    Low_Risk_Rate = 0.5
    High_Risk_Rate = 0.7

    # Initialize population
    X = initsol.copy()

    # Evaluate initial fitness
    fitness = objfun(X)

    # Global best
    best_idx = np.argmin(fitness)
    gbest = X[best_idx, :].copy()
    gbest_fit = fitness[best_idx]

    fitness_curve = []
    start_time = time.time()

    for t in range(Max_iter):
        FRR = High_Risk_Rate - t * ((High_Risk_Rate - Low_Risk_Rate) / Max_iter) #  Compute FRR (Eq. 4)
        for i in range(Npop):
            #  Find Best Nearest Lemur (bnl)
            distances = np.linalg.norm(X - X[i, :], axis=1)
            distances[i] = np.inf
            bnl_idx = np.argmin(distances)
            bnl = X[bnl_idx, :]

            for j in range(dim):
                rand = np.random.rand()
                if rand < FRR:
                    #  Dance-Hub (local exploration)
                    X[i, j] = X[i, j] + abs(X[i, j] - bnl[j]) * (np.random.rand() - 0.5) * 2
                else:
                    #  Leap-Up (global exploitation)
                    X[i, j] = X[i, j] + abs(X[i, j] - gbest[j]) * (np.random.rand() - 0.5) * 2

                #  Boundary check
                X[i, j] = np.clip(X[i, j], xmin[i, j], xmax[i, j])

        # Evaluate fitness
        fitness = objfun(X)

        # Update global best
        current_best_idx = np.argmin(fitness)
        current_best_fit = fitness[current_best_idx]

        if current_best_fit < gbest_fit:
            gbest_fit = current_best_fit
            gbest = X[current_best_idx, :].copy()

        fitness_curve.append(gbest_fit)

    total_time = time.time() - start_time
    return gbest_fit, fitness_curve, gbest, total_time