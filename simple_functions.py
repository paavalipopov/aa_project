import time

import numpy as np
import pandas as pd

import jax


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def beale(x):
    return (
        (1.5 - x[0] + x[0] * x[1]) ** 2
        + (2.25 - x[0] + x[0] * (x[1] ** 2)) ** 2
        + (2.625 - x[0] + x[0] * (x[1] ** 3)) ** 2
    )


def paraboloid(x):
    return x[0] ** 2 + x[1] ** 2


def fwd_ad_iteration(function, params, tangents):
    f_val, jvp = jax.jvp(function, (params,), (tangents,))

    return f_val, jvp * tangents


def rev_ad_terations(function, params, tangents=None):
    f_val, f_vjp = jax.vjp(function, *(params,))

    return f_val, f_vjp((1.0))[0]


func_dict = {"Paraboloid": paraboloid, "Beale": beale, "Rosenbrock": rosenbrock}

ad_dict = {
    "Forward": fwd_ad_iteration,
    "Reverse": rev_ad_terations,
}

for exp in range(100):
    logs = []
    n_iterations = 2000
    for func_name, func in func_dict.items():
        print(f"Running function {func_name}")
        for ad_name, ad_func in ad_dict.items():
            print(f"\tRunning AD {ad_name}")
            lr = 1e-3

            params = np.array([0.5, 0.5])

            log = {
                "Iteration": [0],
                "Time": [0.0],
                "f(x, y)": [func(params)],
                "Function": [func_name],
                "AD mode": [ad_name],
                "x": [params[0]],
                "y": [params[1]],
            }

            start_time = time.perf_counter()
            for iteration in range(1, n_iterations):
                tangents = np.random.rand(2)
                f_val, grad = ad_func(func, params, tangents)

                params = params - lr * grad
                lr *= 0.9999
                current_time = time.perf_counter() - start_time

                log["Iteration"].append(iteration)
                log["Time"].append(current_time)
                log["f(x, y)"].append(f_val)
                log["Function"].append(func_name)
                log["AD mode"].append(ad_name)
                log["x"].append(params[0])
                log["y"].append(params[1])

            print(f"\tElapsed time: {current_time}")
            logs.append(pd.DataFrame(log))

    logs = pd.concat(logs, ignore_index=True)
    logs.to_csv(f"./simple_func_results/no_optim_{exp}.csv")


def adam_update(grad, v, s, iteration, beta1=0.9, beta2=0.999, epsilon=1e-8):
    new_v = beta1 * v + (1 - beta1) * grad
    v_corrected = new_v / (1 - beta1**iteration)

    new_s = beta2 * s + (1 - beta2) * np.square(grad)
    s_corrected = new_s / (1 - beta2**iteration)

    adapted_grad = v_corrected / (np.sqrt(s_corrected) + epsilon)

    return adapted_grad, new_v, new_s


for exps in range(100):
    adam_logs = []
    n_iterations = 2000
    for func_name, func in func_dict.items():
        print(f"Running function {func_name}")
        for ad_name, ad_func in ad_dict.items():
            print(f"\tRunning AD {ad_name}")
            lr = 1e-3

            params = np.array([0.5, 0.5])

            v = np.zeros(params.shape)
            s = np.zeros(params.shape)

            log = {
                "Iteration": [0],
                "Time": [0.0],
                "f(x, y)": [func(params)],
                "Function": [func_name],
                "AD mode": [ad_name],
                "x": [params[0]],
                "y": [params[1]],
            }

            start_time = time.perf_counter()
            for iteration in range(1, n_iterations):
                tangents = np.random.rand(2)
                f_val, grad = ad_func(func, params, tangents)

                grad, v, s = adam_update(grad, v, s, iteration)

                params = params - lr * grad

                current_time = time.perf_counter() - start_time

                log["Iteration"].append(iteration)
                log["Time"].append(current_time)
                log["f(x, y)"].append(f_val)
                log["Function"].append(func_name)
                log["AD mode"].append(ad_name)
                log["x"].append(params[0])
                log["y"].append(params[1])

            print(f"\tElapsed time: {current_time}")
            adam_logs.append(pd.DataFrame(log))

    adam_logs = pd.concat(adam_logs, ignore_index=True)
    adam_logs.to_csv(f"./simple_func_results/adam_optim_{exp}.csv")
