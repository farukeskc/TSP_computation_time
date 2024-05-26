from docplex.mp.model import Model
import numpy as np
import pandas as pd
import random
from itertools import chain


def generate_random_distance_matrix(n):
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n should be a positive integer.")

    lower_triangle = [[random.randint(50,100) if j > i else 0 for j in range(n)] for i in range(n)]

    distance_matrix = [[lower_triangle[j][i] if j < i else lower_triangle[i][j] for j in range(n)] for i in range(n)]

    return distance_matrix

results = []
indexes = []

n_trial = 7

concatenated = chain(range(5, 100, 5), range(100, 140, 10))

for n_cities in concatenated:
    tmp_result = 0

    matrixes = []

    for k in range(n_trial):
        d_matrix = generate_random_distance_matrix(n_cities)

        Cities = range(n_cities)

        model = Model(name='TSP')

        x = model.binary_var_matrix(n_cities, n_cities, name='x')
        u = model.integer_var_list(n_cities, name="u", lb=0)

        obj_expr = model.sum(x[i, j] * d_matrix[i][j] for i in Cities for j in Cities if i != j)

        # Set the objective direction and expression
        model.minimize(obj_expr)

        model.minimize(model.sum(x[i, j] * d_matrix[i][j] for i in Cities for j in Cities if i != j))

        for i in Cities:
            model.add_constraint(model.sum(x[i, j] for j in Cities if i != j) == 1)

        for j in Cities:
            model.add_constraint(model.sum(x[i, j] for i in Cities if i != j) == 1)

        for i in Cities:
            for j in Cities:
                if i >= 2 and j >= 2 and i != j and i <= n_cities and j <= n_cities:
                    model.add_constraint(u[i] - u[j] + 1 <= (n_cities - 1) * (1 - x[i, j]))

        # Constraint 4: u[i] should be greater than or equal to 2
        for i in Cities:
            if i >= 2 and i <= n_cities:
                model.add_constraint(u[i] >= 2)

        # Constraint 5: u[i] should be less than or equal to Ncities
        for i in Cities:
            if i >= 2 and i <= n_cities:
                model.add_constraint(u[i] <= n_cities)

        # Constraint 6: u[1] should be equal to 1
        model.add_constraint(u[1] == 1)

        model.solve()

        tmp_result = tmp_result + model.get_solve_details().time

    tmp_result = tmp_result/n_trial
    results.append(tmp_result)
    indexes.append(n_cities)
    print(n_cities, 'Completed')

print('DONE')

for i, res in enumerate(results):
    print(indexes[i], res)

res = pd.DataFrame([indexes, results])

res.to_csv('results.csv')

