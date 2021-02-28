r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(
        batch_size=32, gamma=0.99, beta=0.5, learn_rate=2e-2, eps=1e-08, num_workers=0, hidden_layers=[128], n_bias=True
        # batch_size=4, gamma=0.99, beta=0.5, learn_rate=7e-4, eps=1e-08, num_workers=0, hidden_layers=[32, 128]

    )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    return hp


def part1_aac_hyperparams():
    hp = dict(
        batch_size=8,
        gamma=0.99,
        beta=0.5,
        delta=5e-5,
        learn_rate=2e-2,
        eps=1e-8,
        num_workers=0,
        hidden_layers=[128],
        n_bias=True
    )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    return hp


part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
