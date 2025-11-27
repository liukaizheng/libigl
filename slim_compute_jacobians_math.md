# Mathematical Explanation of `compute_jacobians` in `igl::slim`

The `compute_jacobians` function calculates the **Deformation Gradient** (or Jacobian matrix) of the piecewise-linear mapping defined by the mesh vertex positions `uv`.

## Mathematical Definition

Let the mapping be $\Phi: \Omega_{\text{ref}} \to \Omega_{\text{deformed}}$, where $\Phi$ is defined by the vertex positions $\mathbf{U} \in \mathbb{R}^{|V| \times d}$ (represented by the variable `uv` in the code).

For each element $e$ (a triangle in 2D or tetrahedron in 3D), the map is linear, so its Jacobian $J_e \in \mathbb{R}^{d \times d}$ is constant over the element. It is defined as the gradient of the mapping:

$$
J_e = \nabla \Phi|_e = \begin{pmatrix} 
\frac{\partial \Phi_1}{\partial x} & \frac{\partial \Phi_1}{\partial y} & \dots \\
\frac{\partial \Phi_2}{\partial x} & \frac{\partial \Phi_2}{\partial y} & \dots \\
\vdots & \vdots & \ddots
\end{pmatrix}
$$

where $\Phi_1, \Phi_2, \dots$ correspond to the $u, v, \dots$ coordinates in the deformed domain.

## Discrete Implementation

The code computes this using precomputed discrete derivative operators. Let $D_x, D_y, D_z$ be sparse matrices (variables `s.Dx`, `s.Dy`, etc.) of size $|F| \times |V|$. When multiplied by a coordinate vector of vertices, they produce the partial derivative of that coordinate for each face.

### Why does Matrix Multiplication produce Derivatives?

This works because of **Piecewise Linear Interpolation**, which is the standard way data is defined on triangle or tetrahedral meshes.

1.  **Linear Interpolation**:
    On any single element (triangle or tetrahedron), a value $u$ at a point $\mathbf{p}$ is defined by interpolating the values at the element's vertices ($u_i, u_j, u_k$) using **Barycentric Coordinates** (or shape functions) $\phi$:

    $$ u(\mathbf{p}) = u_i \phi_i(\mathbf{p}) + u_j \phi_j(\mathbf{p}) + u_k \phi_k(\mathbf{p}) $$

    *   $u_i, u_j, u_k$ are the scalar values stored in your coordinate vector `uv` (vertex positions).
    *   $\phi_i(\mathbf{p})$ is a geometric weight function that is **linear** (it is 1 at vertex $i$ and 0 at the others).

2.  **Taking the Gradient**:
    Since $u_i$ are just constant coefficients (the current vertex positions), the gradient operator $\nabla = [\frac{\partial}{\partial x}, \frac{\partial}{\partial y}, \dots]^T$ passes through them and acts on the shape functions:

    $$ \nabla u(\mathbf{p}) = u_i \nabla \phi_i + u_j \nabla \phi_j + u_k \nabla \phi_k $$

    Because the shape functions $\phi$ are linear, their gradients $\nabla \phi$ are **constant vectors** inside each element. They depend only on the shape (geometry) of the reference triangle/tet, not on the variable values $u$.

3.  **Matrix Form**:
    This sum is effectively a dot product. For the $x$-component of the gradient:

    $$ \frac{\partial u}{\partial x} = u_i \left(\frac{\partial \phi_i}{\partial x}\right) + u_j \left(\frac{\partial \phi_j}{\partial x}\right) + u_k \left(\frac{\partial \phi_k}{\partial x}\right) $$

    This is a linear combination of the vertex values. If you assemble this for every face in the mesh, you get a matrix multiplication:

    $$ \mathbf{g}_x = D_x \cdot \mathbf{u} $$

    Where:
    *   $\mathbf{u}$ is the vector of values at all vertices.
    *   $D_x$ is a sparse matrix where each row corresponds to a face, containing the geometric constants $\frac{\partial \phi}{\partial x}$ for that face's vertices.

### 2D Case (Triangle Mesh)

For a 2D mesh, the Jacobian is a $2 \times 2$ matrix. The code constructs it by flattening the matrix into row-major format or column vectors depending on storage.

The mapping coordinates are vectors $\mathbf{u}$ and $\mathbf{v}$ (columns of `uv`).
The Jacobian matrix for the entire mesh (stacked per face) is constructed as:

$$
J = \begin{pmatrix} D_x \mathbf{u} & D_y \mathbf{u} \\ D_x \mathbf{v} & D_y \mathbf{v} \end{pmatrix}
$$

In the code, this is stored in `s.Ji` as:
$$ \text{Ji} = [\nabla u, \nabla v] = [D_x \mathbf{u}, \quad D_y \mathbf{u}, \quad D_x \mathbf{v}, \quad D_y \mathbf{v}] $$

### 3D Case (Tetrahedral Mesh)

For a 3D mesh, the Jacobian is a $3 \times 3$ matrix. The mapping coordinates are $\mathbf{u}, \mathbf{v}, \mathbf{w}$.

$$
J = \begin{pmatrix} 
D_x \mathbf{u} & D_y \mathbf{u} & D_z \mathbf{u} \\ 
D_x \mathbf{v} & D_y \mathbf{v} & D_z \mathbf{v} \\ 
D_x \mathbf{w} & D_y \mathbf{w} & D_z \mathbf{w} 
\end{pmatrix}
$$

In the code, `s.Ji` stores this flattened as:
$$ \text{Ji} = [\nabla u, \nabla v, \nabla w] $$
specifically:
$$ \text{Ji} = [D_x \mathbf{u}, D_y \mathbf{u}, D_z \mathbf{u}, \quad D_x \mathbf{v}, D_y \mathbf{v}, D_z \mathbf{v}, \quad D_x \mathbf{w}, D_y \mathbf{w}, D_z \mathbf{w}] $$
