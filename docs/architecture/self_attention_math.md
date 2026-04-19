# Issue #13: Self-Attention Mathematical Formulation

## 1. Q, K, V Transformations
Given an input feature map $X \in \mathbb{R}^{C \times H \times W}$, we first flatten the spatial dimensions to get $X \in \mathbb{R}^{C \times N}$ where $N = H \times W$.

### Linear Projections:
- **Query (Q):** $Q = W_q X$, where $W_q \in \mathbb{R}^{d_k \times C}$.
- **Key (K):** $K = W_k X$, where $W_k \in \mathbb{R}^{d_k \times C}$.
- **Value (V):** $V = W_v X$, where $W_v \in \mathbb{R}^{C \times C}$.

In our implementation, $d_k = C / 8$ to reduce computational cost.

## 2. Attention Map Computation
The attention score $s_{ij}$ between position $i$ and position $j$ is computed using the dot product:

$$ s_{ij} = Q_i^T K_j $$

The attention map $A$ is then obtained via softmax:

$$ A_{ij} = \frac{\exp(s_{ij})}{\sum_{k=1}^N \exp(s_{ik})} $$

## 3. Context Vector and Residual Connection
The output of the attention block $O$ is:

$$ O = A V^T $$

Finally, the residual connection with a learnable scale $\gamma$ is applied:

$$ Y = \gamma \cdot O + X $$

## 4. Complexity Analysis
- **Memory:** The attention matrix $A$ has size $N \times N$. For $256 \times 256$ images ($N=65536$), this would require ~16GB of memory, which is why we place it after the $64 \times 64$ stage ($N=4096$).
- **FLOPs:** $O(N^2 \cdot C)$.
