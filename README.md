# Implementation of Value Iteration for Optimal Policy Computation using Gymnasium

## Aim

To implement the **Value Iteration algorithm** for solving a finite **Markov Decision Process (MDP)** using a customized **Gymnasium FrozenLake-v1 environment**, and to compute the **optimal state-value function** and **optimal policy** using the Bellman Optimality Equation.

---

## Problem Statement

Implement the **Value Iteration algorithm** to determine the optimal value of each state and the best action to take from every state in a customized FrozenLake environment.

The algorithm uses the **Bellman Optimality Equation**:

**V*(s) = maxₐ Σₛ′ P(s′|s,a) [R(s,a,s′) + γV*(s′)]**

The algorithm repeatedly updates the value of each state until the value function converges. After convergence, the optimal policy is obtained by selecting the action with the maximum expected value.

---

## Software Requirements

### Hardware Requirements

- Computer/Laptop
- Minimum 4 GB RAM
- Internet connection

### Software Requirements

- Python 3.x
- Google Colab / Jupyter Notebook
- Gymnasium
- NumPy

Install Gymnasium:

```python
!pip install gymnasium
```

Import the required libraries:

```python
import gymnasium as gym
import numpy as np
```

---

## Environment Description

**FrozenLake-v1** is a grid-world environment provided by Gymnasium. The agent must navigate across a frozen lake from a starting position to a goal position while avoiding holes.

In this experiment, a customized **5 × 5 FrozenLake environment** is used.

The environment contains **25 states** arranged in a 5 × 5 grid.

### Environment Configuration

```python
env_desc1 = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

env1 = gym.make(
    "FrozenLake-v1",
    desc=env_desc1,
    is_slippery=True
)
```

### Grid Representation

```text
S F F F
F H F H
F F F H 
H F F G
```

### Symbols

- **S** → Starting position
- **F** → Frozen/safe tile
- **H** → Hole
- **G** → Goal

In this environment:

- **Start (S)** is at the top-right corner.
- **Goal (G)** is at the bottom-left corner.

Since `is_slippery=True`, the environment is stochastic, meaning the agent may not always move exactly in the intended direction.

### Available Actions

| Action | Symbol | Description |
|---|---|---|
| 0 | L | Move Left |
| 1 | D | Move Down |
| 2 | R | Move Right |
| 3 | U | Move Up |

The objective is to find an optimal policy that guides the agent from the starting state to the goal while avoiding holes and maximizing the expected reward.

---

## MDP Representation

The FrozenLake environment can be represented as a Markov Decision Process:

$$
M = (S,A,P,R,\gamma)
$$

where:

- **S** → Set of states
- **A** → Set of actions
- **P** → Transition probability
- **R** → Reward function
- **γ** → Discount factor

### MDP Components

| Component | Description |
|---|---|
| Number of States | 25 |
| Grid Size | 5 × 5 |
| Number of Actions | 4 |
| Start State | Top-right |
| Goal State | Bottom-left |
| Environment | FrozenLake-v1 |
| Slippery | True |
| Type | Stochastic MDP |
| Discount Factor | 0.99 |

---

## Theory

### Value Iteration

Value Iteration is a dynamic programming algorithm used to find the optimal value function and optimal policy for a finite Markov Decision Process (MDP).

The algorithm starts by assigning an initial value of zero to all states.

**V₀(s) = 0**

At each iteration, the value of every state is updated using the Bellman Optimality Equation:

**Vₖ₊₁(s) = maxₐ Σₛ′ P(s′|s,a) [R(s,a,s′) + γVₖ(s′)]**

where:

- **Vₖ(s)** → Value of state s at iteration k
- **s** → Current state
- **s′** → Next state
- **a** → Possible action
- **P(s′|s,a)** → Probability of reaching next state s′ from state s after taking action a
- **R(s,a,s′)** → Reward received after taking action a
- **γ** → Discount factor
- **maxₐ** → Selects the action with the maximum expected value

### Convergence

Value Iteration repeatedly updates the values of all states until the values become stable.

The maximum change in the value function is calculated as:

**Δ = maxₛ |V_new(s) - V(s)|**

The algorithm stops when:

**Δ < θ**

In this experiment:

- **γ = 0.99**
- **θ = 1e-8**

A smaller value of θ provides a more accurate convergence condition.

### Optimal Policy

After the value function converges, the optimal policy is obtained by selecting the action with the highest expected value for each state.

**π*(s) = argmaxₐ Σₛ′ P(s′|s,a) [R(s,a,s′) + γV(s′)]**

The resulting policy tells the agent which action to take from each state.

For this experiment, the possible actions are:

| Action | Symbol | Meaning |
|---|---|---|
| 0 | L | Move Left |
| 1 | D | Move Down |
| 2 | R | Move Right |
| 3 | U | Move Up |

Since the FrozenLake environment is configured with `is_slippery=True`, the agent considers the probability of different transitions while calculating the expected value of each action.

---

## Algorithm

1. Create the customized 5 × 5 FrozenLake environment.
2. Initialize the value function `V` with zeros.
3. Set the discount factor `gamma`.
4. Set the convergence threshold `theta`.
5. For each state, calculate the expected value of every possible action.
6. Select the action having the maximum expected value.
7. Update the value of the state.
8. Calculate the maximum change `delta`.
9. Repeat until `delta < theta`.
10. Extract the optimal policy using `argmax`.
11. Display the number of iterations.
12. Display the optimal state-value function as a 5 × 5 matrix.
13. Display the optimal policy using action symbols.

---

# Python Program

## Create FrozenLake Environment

```python
# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

env_desc1 = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

env1 = gym.make(
    "FrozenLake-v1",
    desc=env_desc1,
    is_slippery=True
)
```

## Value Iteration Algorithm

```python
# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------

def value_iteration(env1, gamma=0.99, theta=1e-8):
    """
    Performs Value Iteration and returns
    the optimal value function, policy,
    and number of iterations.
    """

    n_states = env1.observation_space.n
    n_actions = env1.action_space.n

    # Initialize value function
    V = np.zeros(n_states)

    iteration = 0

    while True:

        delta = 0

        for s in range(n_states):

            action_values = []

            for a in range(n_actions):

                value = 0

                # Get transition information
                for probability, next_state, reward, terminated in env1.unwrapped.P[s][a]:

                    value += probability * (
                        reward +
                        gamma * V[next_state] * (not terminated)
                    )

                action_values.append(value)

            # Bellman Optimality Equation
            new_value = max(action_values)

            delta = max(
                delta,
                abs(new_value - V[s])
            )

            V[s] = new_value

        iteration += 1

        # Check convergence
        if delta < theta:
            break

    # -------------------------------------------------
    # Extract Optimal Policy
    # -------------------------------------------------

    policy = np.zeros(n_states, dtype=int)

    for s in range(n_states):

        action_values = []

        for a in range(n_actions):

            value = 0

            for probability, next_state, reward, terminated in env1.unwrapped.P[s][a]:

                value += probability * (
                    reward +
                    gamma * V[next_state] * (not terminated)
                )

            action_values.append(value)

        policy[s] = np.argmax(action_values)

    return V, policy, iteration
```

## Run Value Iteration

```python
# -------------------------------------------------
# Run Value Iteration
# -------------------------------------------------

V, policy, iterations = value_iteration(env1)
```

## Display Output

```python
# -------------------------------------------------
# Display Output
# -------------------------------------------------

print("Name: Niralya J")
print("Register Number: 212224230188")
print("Value Iteration Completed")
print("Number of Iterations:", iterations)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(5, 5), 4))

action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

policy_grid = np.array(
    [action_symbols[action] for action in policy]
).reshape(5, 5)

print("\nOptimal Policy:")
print(policy_grid)

env1.close()
```

---

## Output

The program produces the following output format:

<img width="537" height="362" alt="Screenshot 2026-08-05 111903" src="https://github.com/user-attachments/assets/3eacc883-279d-4406-8863-b227eba8917b" />

## New enviroment description

<img width="653" height="398" alt="Screenshot 2026-08-05 112228" src="https://github.com/user-attachments/assets/883876b7-cd8e-4146-b5f2-c9895b2e4eac" />


---

## Result

The Value Iteration algorithm was successfully implemented using the customized 5 × 5 FrozenLake-v1 environment with env1. The algorithm converged to the optimal state-value function and generated the corresponding optimal policy using the Bellman Optimality Equation. The optimal policy provides the best actions for navigating from the top-right starting state to the bottom-left goal state while avoiding holes.


## Inference

The two executions show that Value Iteration is influenced by the transition behavior of the environment.

The first execution converged in 7 iterations with higher state values, while the second required 252 iterations and produced lower, more varied state values. 

The optimal policies also differed due to changes in the expected outcomes of actions. 

This demonstrates that Value Iteration effectively learns an optimal policy by considering transition probabilities, rewards, and future state values in a stochastic MDP.
