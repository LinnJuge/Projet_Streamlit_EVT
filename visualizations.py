import matplotlib.pyplot as plt
import seaborn as sns

def plot_returns_distribution(data, confidence=0.95, weights=None, show_var_param=True, show_var_hist=True, show_var_mc=True):
    """
    Affiche la distribution des rendements avec la VaR et la CVaR.
    """
    data = get_portfolio_returns(data, weights)
    plt.figure(figsize=(12,6))
    sns.histplot(data, bins=50, kde=True, color="blue")

    var_param = calculate_var(data, confidence) if show_var_param else None
    var_hist = var_historique(data, confidence) if show_var_hist else None
    var_mc = var_monte_carlo(data, confidence) if show_var_mc else None
    cvar = calculate_cvar(data, confidence)

    if show_var_param:
        plt.axvline(-var_param, color='purple', linestyle='--', label=f'VaR Paramétrique: {var_param:.4f}')
    if show_var_hist:
        plt.axvline(-var_hist, color='red', linestyle='--', label=f'VaR Historique: {var_hist:.4f}')
    if show_var_mc:
        plt.axvline(-var_mc, color='green', linestyle='--', label=f'VaR Monte Carlo: {var_mc:.4f}')

    plt.axvline(-cvar, color='black', linestyle='-', linewidth=2, label=f'CVaR: {cvar:.4f}')
    plt.legend()
    plt.title("Distribution des Rendements avec VaR et CVaR")
    plt.show()
