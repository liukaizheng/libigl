# Mathematical Explanation of `build_linear_system` in `igl::slim`

The `build_linear_system` function constructs the linear system used in the "local-global" solver of the SLIM (Scalable Locally Injective Mappings) algorithm. SLIM minimizes a distortion energy by iteratively solving a quadratic proxy energy.

## 1. The Optimization Problem

The goal of SLIM is to minimize a total distortion energy $E(V)$ defined over the mesh:
$$ E(V) = \sum_{e \in \text{elements}} \text{vol}_e \cdot \Psi(J_e) $$
where $J_e$ is the deformation gradient (Jacobian) of element $e$, and $\Psi$ is a distortion metric (e.g., Symmetric Dirichlet, ARAP, Conformal).

To minimize this efficiently, SLIM uses a **proximal operator** approach (specifically, a Gauss-Newton-like or Alternating Optimization Strategy). In each iteration, it constructs a quadratic proxy energy that approximates the original energy around a target frame $R_e$:

$$ E_{\text{proxy}}(U) = \sum_{e} \text{vol}_e \| W_e (J_e(U) - R_e) \|_F^2 + \frac{\lambda}{2} \| U - U_{\text{old}} \|^2 $$

Where:
*   $U$ are the unknown vertex positions.
*   $J_e(U)$ is the Jacobian of the deformation as a linear function of $U$.
*   $R_e$ is the "closest rotation" or target frame derived from the current deformation.
*   $W_e$ is a weighting matrix (derived from the Hessian of the distortion energy $\Psi$).
*   $\text{vol}_e$ is the volume/area of the element.
*   $\lambda$ is a proximal regularization weight (`proximal_p`).

## 2. Linear System Formulation

The minimization of the quadratic proxy energy leads to a linear system of the form:
$$ L \cdot U = \text{rhs} $$


### 2.1 Detailed Derivation as Weighted Least Squares

The problem solved in each iteration is a **Weighted Least Squares (WLS)** optimization. Let's break down why.

We want to find $U$ that minimizes the difference between our *actual* weighted deformation ($W J(U)$) and our *target* weighted deformation ($W R$).

$$ \min_U \sum_{e} \text{vol}_e \| W_e J_e(U) - W_e R_e \|_F^2 $$

This can be rewritten as a global matrix norm. Let's define:
*   $\mathbf{A}$: A large sparse matrix that computes Jacobians from positions $U$. Specifically, it computes the *weighted* Jacobians.
    $$ \mathbf{A} U \approx \text{vector of } \{ W_e J_e \}_e $$
*   $\mathbf{b}$: A vector representing the target deformations.
    $$ \mathbf{b} \approx \text{vector of } \{ W_e R_e \}_e $$
*   $\mathbf{M}$: A diagonal matrix containing the volume/area weights ($\text{vol}_e$) for every entry.

The energy becomes:
$$ E(U) = (\mathbf{A} U - \mathbf{b})^T \mathbf{M} (\mathbf{A} U - \mathbf{b}) + \frac{\lambda}{2} \| U - U_{\text{old}} \|^2 $$

To find the minimum, we take the derivative with respect to $U$ and set it to zero:
$$ \nabla_U E = 2 \mathbf{A}^T \mathbf{M} (\mathbf{A} U - \mathbf{b}) + \lambda (U - U_{\text{old}}) = 0 $$

Rearranging terms to group $U$ on the left side:
$$ \mathbf{A}^T \mathbf{M} \mathbf{A} U + \lambda U = \mathbf{A}^T \mathbf{M} \mathbf{b} + \lambda U_{\text{old}} $$
$$ (\mathbf{A}^T \mathbf{M} \mathbf{A} + \lambda I) U = \mathbf{A}^T \mathbf{M} \mathbf{b} + \lambda U_{\text{old}} $$

This exactly matches the code implementation:
*   **LHS Matrix ($L$)**: `s.AtA` (which is $A^T M A$) + `s.proximal_p * id_m` ($\lambda I$).
*   **RHS Vector**: `s.rhs` calculated as $A^T M b + \lambda U_{\text{old}}$.

### 2.2 The Matrix $A$ (`slim_buildA`)

The sparse matrix $A$ represents the linear operator that maps vertex positions $U$ to the **weighted Jacobian components**.

$$ A \cdot U = \text{vec}(\{ W_e J_e \}_e) $$

For a triangle mesh (2D), the Jacobian $J_e$ is $2 \times 2$, and the weight matrix $W_e$ is effectively $2 \times 2$.
$$ (W_e J_e) = \begin{pmatrix} W_{00} & W_{01} \\ W_{10} & W_{11} \end{pmatrix} \begin{pmatrix} u_x & u_y \\ v_x & v_y \end{pmatrix} = \begin{pmatrix} W_{00} u_x + W_{01} v_x & W_{00} u_y + W_{01} v_y \\ W_{10} u_x + W_{11} v_x & W_{10} u_y + W_{11} v_y \end{pmatrix} $$

The matrix $A$ is constructed by stacking these 4 components for all faces.
*   **Rows**: The matrix has $d^2 \cdot |F|$ rows (where $d$ is dimension).
    *   Block 0 (Rows $0 \to |F|$): Represents $(W J)_{00}$
    *   Block 1 (Rows $|F| \to 2|F|$): Represents $(W J)_{01}$
    *   ...etc.
*   **Columns**: The matrix has $d \cdot |V|$ columns, corresponding to flattened vertex coordinates $[u_1 \dots u_n, v_1 \dots v_n]$.

**In the code (`slim_buildA`):**
*   Terms involving $D_x$ (partial derivative wrt x) contribute to columns representing $u_x$ and $v_x$.
    *   $W_{00} u_x + W_{01} v_x$ (Block 0)
    *   $W_{10} u_x + W_{11} v_x$ (Block 2)
*   Terms involving $D_y$ contribute to columns representing $u_y$ and $v_y$.
    *   $W_{00} u_y + W_{01} v_y$ (Block 1)
    *   $W_{10} u_y + W_{11} v_y$ (Block 3)

### 2.2 The Matrix $L$ (`build_linear_system`)

$$ L = A^T \cdot \text{diag}(\text{Mesh Area Weights}) \cdot A + \text{Proximal Term} $$

*   `s.WGL_M`: Vector containing element areas/volumes, repeated for each block of $A$.
*   `s.proximal_p`: The scalar $\lambda$.
*   `id_m`: Identity matrix.

This creates the Hessian of the energy: $L = \nabla^2 E_{\text{proxy}}$.

### 2.3 The RHS Vector (`buildRhs`)

The right-hand side represents the target force pulling the vertices towards the optimal Jacobian configuration.
$$ \text{rhs} = A^T \mathbf{M}_{area} \text{vec}(\{ W_e R_e \}_e) $$

The vector `f_rhs` is constructed to match the structure of $A \cdot U$. It contains the flattened components of the weighted target frames $W_e R_e$:
*   Block 0: $(W R)_{00} = W_{00} R_{00} + W_{01} R_{10}$
*   Block 1: $(W R)_{01} = W_{00} R_{01} + W_{01} R_{11}$
*   etc.

Finally, the proximal regularization term is added: $\text{rhs} \leftarrow \text{rhs} + \lambda U_{\text{old}}$.

## 3. Soft Constraints

The function `add_soft_constraints` modifies $L$ and $\text{rhs}$ to weakly enforce positional constraints (anchor points).
$$ E_{\text{total}} = E_{\text{proxy}} + p_{\text{soft}} \sum_{k \in \text{anchors}} \| u_k - u_k^{\text{target}} \|^2 $$
This adds $p_{\text{soft}}$ to the diagonal of $L$ at the indices of constrained vertices and updates the RHS with $p_{\text{soft}} u^{\text{target}}$.
