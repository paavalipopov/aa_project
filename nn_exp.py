import time
import glob

import numpy as np
import pandas as pd

from random import randint

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax import random
from jax.scipy.special import logsumexp

import mnist

x_train, t_train, x_test, t_test = mnist.load()
# x_train, t_train, x_test, t_test = x_train[:2000], t_train[:2000], x_test[:2000], t_test[:2000]
num_pixels = x_train.shape[1]
num_labels = np.unique(t_train).shape[0]


# A helper function to randomly initialize weights and biases
# for a dense neural network layer
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_network_params(sizes, key, scale=1e-2):
    keys = random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k, scale)
        for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


def random_tangent(m, n, scale):
    return scale * np.random.rand(n, m), scale * np.random.rand(n)


def init_tangents(sizes, scale=1.0):
    return [random_tangent(m, n, scale) for m, n in zip(sizes[:-1], sizes[1:])]


def zero_vects(m, n):
    return np.zeros((n, m)), np.zeros((n))


def init_zeros(sizes):
    return [zero_vects(m, n) for m, n in zip(sizes[:-1], sizes[1:])]


class Batcher:
    def __init__(self, data, labels, batch_size) -> None:
        self.data = data
        self.labels = labels
        self.batch_size = batch_size

        self.counter = 0

    def reset(self):
        self.counter = 0

    def get_all(self):
        return self.data, self.labels

    def get_batch(self, randomomized=False):
        if randomomized and self.counter == 0:
            p = np.random.permutation(self.labels.shape[0])
            self.data = self.data[p]
            self.labels = self.labels[p]

        if self.batch_size * self.counter >= self.data.shape[0]:
            self.counter = 0
            return None, None

        elif self.batch_size * (self.counter + 1) < self.data.shape[0]:
            data, labels = (
                self.data[
                    self.counter
                    * self.batch_size : (self.counter + 1)
                    * self.batch_size
                ],
                self.labels[
                    self.counter
                    * self.batch_size : (self.counter + 1)
                    * self.batch_size
                ],
            )
            self.counter += 1
            return data, labels
        else:
            data, labels = (
                self.data[self.counter * self.batch_size :],
                self.labels[self.counter * self.batch_size :],
            )
            self.counter += 1
            return data, labels


def relu(x):
    return jnp.maximum(0, x)


def predict(params, image):
    # per-example predictions
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)


batched_predict = vmap(predict, in_axes=(None, 0))


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def accuracy(params, images, targets):
    # target_class = jnp.argmax(targets, axis=1)
    target_class = targets
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)


def loss(params, images, targets):
    targets = one_hot(targets, num_labels)
    preds = batched_predict(params, images)
    return -jnp.mean(preds * targets)


def report(params, images, targets):
    preds = batched_predict(params, images)
    predicted_class = jnp.argmax(preds, axis=1)
    oh_targets = one_hot(targets, num_labels)

    return jnp.mean(predicted_class == targets), -jnp.mean(preds * oh_targets)


@jit
def fwd_ad_iteration(x, y, params, tangents):
    isolated_loss = lambda params: loss(params, x, y)

    t_0 = time.perf_counter()
    f_val, jvp = jax.jvp(isolated_loss, (params,), (tangents,))
    time_elapsed = time.perf_counter() - t_0

    return f_val, [(jvp * dw, jvp * db) for (dw, db) in tangents], time_elapsed


@jit
def rev_ad_iteration(x, y, params, tangents=None):
    isolated_loss = lambda params: loss(params, x, y)

    t_0 = time.perf_counter()
    f_val, vjp = jax.vjp(isolated_loss, *(params,))
    time_elapsed = time.perf_counter() - t_0

    return f_val, vjp((1.0))[0], time_elapsed


@jit
def update_params(params, grads, lr):
    return [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(params, grads)]


@jit
def adam_update(grad, v, s, iteration, beta1=0.9, beta2=0.999, epsilon=1e-8):
    new_v = [
        (beta1 * dwv + (1 - beta1) * dw, beta1 * dbv + (1 - beta1) * db)
        for (dwv, dbv), (dw, db) in zip(v, grad)
    ]
    v_corrected = [
        (nvw / (1 - beta1**iteration), nvb / (1 - beta1**iteration))
        for (nvw, nvb) in new_v
    ]

    new_s = [
        (
            beta2 * dws + (1 - beta2) * jnp.square(dw),
            beta2 * dbs + (1 - beta2) * jnp.square(db),
        )
        for (dws, dbs), (dw, db) in zip(s, grad)
    ]
    s_corrected = [
        (nsw / (1 - beta2**iteration), nsb / (1 - beta2**iteration))
        for (nsw, nsb) in new_s
    ]

    adapted_grad = [
        (vcw / (jnp.sqrt(scw) + epsilon), vcb / (jnp.sqrt(scb) + epsilon))
        for (vcw, vcb), (scw, scb) in zip(v_corrected, s_corrected)
    ]

    return adapted_grad, new_v, new_s


ad_dict = {
    "Forward": fwd_ad_iteration,
    "Reverse": rev_ad_iteration,
}
optimization = [
    "No optimization",
    "Adam",
]


layer_sizes = [784, 10]
num_epochs = 20
batch_size = 100

logs = []
epoch_logs = []
for exp in range(100):
    log = {
        "Iteration": [],
        "Time": [],
        "Test accuracy": [],
        "Test loss": [],
        "Train accuracy": [],
        "Train loss": [],
        "AD mode": [],
        "Optimization": [],
    }
    epoch_log = {
        "Epoch": [],
        "Time": [],
        "Test accuracy": [],
        "Test loss": [],
        "Train accuracy": [],
        "Train loss": [],
        "AD mode": [],
        "Optimization": [],
    }

    for opt in optimization:
        for ad_name, ad_func in ad_dict.items():
            lr = 1e-3
            params = init_network_params(layer_sizes, random.PRNGKey(0))
            v = init_zeros(layer_sizes)
            s = init_zeros(layer_sizes)

            time_elapsed = 0.0
            update_iter = 0

            batcher = Batcher(x_train, t_train, batch_size)

            train_acc, train_loss = report(params, x_train, t_train)
            test_acc, test_loss = report(params, x_test, t_test)

            log["Iteration"].append(0)
            log["Time"].append(0)
            log["Test accuracy"].append(test_acc)
            log["Test loss"].append(test_loss)
            log["Train accuracy"].append(train_acc)
            log["Train loss"].append(train_loss)
            log["AD mode"].append(ad_name)
            log["Optimization"].append(opt)

            epoch_log["Epoch"].append(0)
            epoch_log["Time"].append(0)
            epoch_log["Test accuracy"].append(test_acc)
            epoch_log["Test loss"].append(test_loss)
            epoch_log["Train accuracy"].append(train_acc)
            epoch_log["Train loss"].append(train_loss)
            epoch_log["AD mode"].append(ad_name)
            epoch_log["Optimization"].append(opt)

            for epoch in range(num_epochs):
                start_time = time.time()
                x, y = batcher.get_batch(randomomized=True)

                while x is not None and y is not None:
                    print(f"Iteration {update_iter}")
                    tangents = init_tangents(layer_sizes)

                    x = jnp.reshape(x, (len(x), num_pixels))

                    loss_v, grad, delta_t = ad_func(x, y, params, tangents)

                    if opt == "Adam":
                        grad, v, s = adam_update(grad, v, s, update_iter + 1)
                    params = update_params(params, grad, lr)

                    if opt != "Adam":
                        lr *= 0.9999

                    time_elapsed += delta_t

                    train_acc, train_loss = report(params, x_train, t_train)
                    test_acc, test_loss = report(params, x_test, t_test)

                    log["Iteration"].append(update_iter + 1)
                    log["Time"].append(time_elapsed)
                    log["Test accuracy"].append(test_acc)
                    log["Test loss"].append(test_loss)
                    log["Train accuracy"].append(train_acc)
                    log["Train loss"].append(train_loss)
                    log["AD mode"].append(ad_name)
                    log["Optimization"].append(opt)

                    x, y = batcher.get_batch(randomomized=True)

                    update_iter += 1

                print(f"Epoch {epoch}")

                epoch_log["Epoch"].append(epoch + 1)
                epoch_log["Time"].append(time_elapsed)
                epoch_log["Test accuracy"].append(test_acc)
                epoch_log["Test loss"].append(test_loss)
                epoch_log["Train accuracy"].append(train_acc)
                epoch_log["Train loss"].append(train_loss)
                epoch_log["AD mode"].append(ad_name)
                epoch_log["Optimization"].append(opt)

    logs.append(pd.DataFrame(log))
    epoch_logs.append(pd.DataFrame(epoch_log))

    pd.DataFrame(log).to_csv(f"./nn_results/log_{exp}.csv")
    pd.DataFrame(epoch_log).to_csv(f"./nn_results/exp_log_{exp}.csv")

