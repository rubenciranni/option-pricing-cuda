import math
import numpy as np
import matplotlib.pyplot as plt

def american_put_binomial(S, K, T, r, sigma, q, n):
    """
    American Put via binomial tree (pseudocode-style weights p0/p1).

    Returns:
        price            : float, option price at root
        values           : list of lists, option value at each node [time j][i]
        exercise_mask    : list of lists of bool, True if node value equals intrinsic (exercise taken)
        stock_prices     : list of lists, S at each node [time j][i]
    """
    dt = T / n
    up = math.exp(sigma * math.sqrt(dt))

    # p0/p1 per your pseudocode (equivalent to risk-neutral/discounted form with d=1/u)
    disc = math.exp(-r * dt)
    p0 = (up * math.exp(-q * dt) - disc) / (up**2 - 1.0)
    p1 = disc - p0

    # Tree containers
    values = [ [0.0]*(j+1) for j in range(n+1) ]
    exercise_mask = [ [False]*(j+1) for j in range(n+1) ]
    stock_prices = [ [0.0]*(j+1) for j in range(n+1) ]

    # Underlying prices at each node: S * up^(2*i - j)
    for j in range(n+1):
        for i in range(j+1):
            stock_prices[j][i] = S * (up ** (2*i - j))

    # Terminal payoffs at maturity (j = n)
    for i in range(n+1):
        payoff = K - stock_prices[n][i]
        if payoff < 0:
            payoff = 0.0
        values[n][i] = payoff
        # Mark red if zero at maturity or if it's intrinsic (here terminal always intrinsic anyway)
        exercise_mask[n][i] = (payoff == 0.0) or (payoff > 0.0)

    # Backward induction
    # p[j-1][i] = p0 * p[j][i+1] + p1 * p[j][i]
    # exercise = K - S * up^(2*i - j)
    for j in range(n-1, -1, -1):
        for i in range(j+1):
            continuation = p0 * values[j+1][i+1] + p1 * values[j+1][i]
            intrinsic = K - stock_prices[j][i]
            if intrinsic < 0:
                intrinsic = 0.0

            # American choice
            v = max(continuation, intrinsic)
            values[j][i] = v

            # Red if we exercise (value equals intrinsic) OR value == 0. Blue otherwise.
            exercised = (abs(v - intrinsic) < 1e-12) and (intrinsic > 0 or v == 0)
            zero_val = (abs(v) < 1e-12)
            exercise_mask[j][i] = exercised or zero_val

    return values[0][0], values, exercise_mask, stock_prices


def plot_binomial_tree(values, exercise_mask, title="American Put Binomial Tree"):
    """
    Visualize the tree:
      - nodes colored red if exercise_mask True (exercise or zero value), else blue
      - annotate node with value
    """
    n = len(values) - 1
    fig, ax = plt.subplots(figsize=(max(8, n*0.9), max(6, n*0.7)))

    # Coordinates: x = time step j, y = centered index so tree looks balanced
    # We'll space y so that for each time j, the i indices are vertically centered.
    xs, ys, cs, texts = [], [], [], []
    for j in range(n+1):
        # center around 0
        y_base = -j/2
        for i in range(j+1):
            x = j
            y = y_base + i
            xs.append(x); ys.append(y)
            cs.append('red' if exercise_mask[j][i] else 'blue')
            texts.append(f"{values[j][i]:.2f}")

    # Draw edges: from (j,i) to (j+1,i) and (j+1,i+1)
    for j in range(n):
        y_base_curr = -j/2
        y_base_next = -(j+1)/2
        for i in range(j+1):
            x0, y0 = j, y_base_curr + i
            # down child (i)
            x1, y1 = j+1, y_base_next + i
            ax.plot([x0, x1], [y0, y1], linewidth=1, alpha=0.5)
            # up child (i+1)
            x2, y2 = j+1, y_base_next + (i+1)
            ax.plot([x0, x2], [y0, y2], linewidth=1, alpha=0.5)

    # Draw nodes
    ax.scatter(xs, ys, s=80, c=cs, zorder=3, edgecolors='k', linewidths=0.5)

    # Annotate values
    for x, y, t in zip(xs, ys, texts):
        ax.text(x, y + 0.08, t, ha='center', va='bottom', fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("Time step (j)")
    ax.set_ylabel("Node index (centered)")
    ax.set_xticks(range(n+1))
    ax.grid(True, linestyle='--', alpha=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example parameters (tweak as you like)
    S = 100.0     # spot
    K = 100.0     # strike
    T = 1.0       # years
    r = 0.05      # risk-free rate
    sigma = 0.2   # volatility
    q = 0.0       # dividend yield
    n = 10        # steps

    price, values, mask, _ = american_put_binomial(S, K, T, r, sigma, q, n)
    print(f"American put price (n={n}): {price:.4f}")
    plot_binomial_tree(values, mask, title=f"American Put Binomial Tree (n={n})")
