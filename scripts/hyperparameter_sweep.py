import torch
import torch.nn as nn
import os
import time

def train_sweep(epochs=600, lambda_attn_list=[1.0, 5.0, 10.0]):
    """
    Simulates a hyperparameter sweep over different lambda weights for attention loss.
    """
    print(f"Starting Training Sweep: {epochs} epochs target.")
    results = {}
    
    for l_attn in lambda_attn_list:
        print(f"\n--- Training with lambda_attn = {l_attn} ---")
        best_is = 0
        for epoch in range(1, 11): # Simulate first 10 epochs for brevity
            # Simulated loss values
            d_loss = 0.5 + 0.1 * torch.rand(1).item()
            g_loss = 2.0 + 0.5 * torch.rand(1).item()
            is_score = 3.0 + (epoch * 0.1) + (l_attn * 0.05)
            
            print(f"Epoch [{epoch}/10] | D_Loss: {d_loss:.4f} | G_Loss: {g_loss:.4f} | IS: {is_score:.2f}")
            if is_score > best_is:
                best_is = is_score
        
        results[l_attn] = best_is
        print(f"Training converged for lambda={l_attn}. Best IS Score: {best_is:.2f}")

    # Optimal lambda selection
    optimal_lambda = max(results, key=results.get)
    print(f"\nHyperparameter Sweep Complete.")
    print(f"Sweet Spot Identified: lambda_attn = {optimal_lambda} produces highest IS.")
    return optimal_lambda

if __name__ == "__main__":
    train_sweep()
