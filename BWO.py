import numpy as np
import time


def BWO(initsol, objfun, xmin, xmax, Max_iter):
    # Black Widow Optimization (BWO)
    # Parameters (can be tuned)
    pop_size = initsol.shape[0]
    dim = initsol.shape[1]
    Pp = 0.6          # Procreation rate
    CR = 0.44         # Cannibalism rate
    MR = 0.1          # Mutation rate

    # Initialize population
    pop = initsol.copy()
    fitness = objfun(pop)

    # Best solution
    best_idx = np.argmin(fitness)
    best_sol = pop[best_idx].copy()
    best_fit = fitness[best_idx]

    convergence = []
    start_time = time.time()
    for t in range(Max_iter):
        new_pop = []

        # PROCREATION (Eq. 3)
        num_pairs = int(Pp * pop_size / 2)

        for _ in range(num_pairs):
            i, j = np.random.choice(pop_size, 2, replace=False)
            Xi, Xj = pop[i], pop[j]

            alpha = np.random.rand()

            # Offspring generation
            Y1 = alpha * Xi + (1 - alpha) * Xj
            Y2 = alpha * Xj + (1 - alpha) * Xi

            new_pop.append(Y1)
            new_pop.append(Y2)

        new_pop = np.array(new_pop)

        # MUTATION
        num_mut = int(MR * len(new_pop))
        for i in range(num_mut):
            idx = np.random.randint(len(new_pop))
            mutation = np.random.uniform(-1, 1, dim)
            new_pop[idx] = new_pop[idx] + mutation * (xmax[0] - xmin[0])

        # Boundary control
        new_pop = np.clip(new_pop, xmin[0], xmax[0])

        # FITNESS EVALUATION
        new_fitness = objfun(new_pop)

        # CANNIBALISM (SELECTION)
        total_pop = np.vstack((pop, new_pop))
        total_fit = np.hstack((fitness, new_fitness))

        # Sort based on fitness (minimization)
        sorted_idx = np.argsort(total_fit)

        survivors = int((1 - CR) * pop_size)
        survivors = max(survivors, 2)

        pop = total_pop[sorted_idx[:survivors]]

        # Refill population if needed
        while pop.shape[0] < pop_size:
            rand_sol = np.random.uniform(xmin[0], xmax[0], dim)
            pop = np.vstack((pop, rand_sol))

        fitness = objfun(pop)

        # Update best
        current_best_idx = np.argmin(fitness)
        current_best_fit = fitness[current_best_idx]

        if current_best_fit < best_fit:
            best_fit = current_best_fit
            best_sol = pop[current_best_idx].copy()

        convergence.append(best_fit)

    total_time = time.time() - start_time
    return best_fit, convergence, best_sol, total_time

