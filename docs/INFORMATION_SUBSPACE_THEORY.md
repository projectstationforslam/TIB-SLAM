# Degradation-aware information-subspace update

## Definition

Let the sparse frontend provide a tangent-space estimate with covariance
`Sigma_s`, and let `R` be a positive-semidefinite model for the independent
RGB-D/geometry update noise. Define

`S = Sigma_s + R` and `G = S^(-1/2) Sigma_s S^(-1/2)`.

For the eigendecomposition `G = U diag(g_i) U^T`, we select a binary mask
`p_i` using a degradation-adaptive gain threshold and a directional NIS gate.
The applied tangent update is

`delta_plus = A delta`,

`A = S^(1/2) U diag(p_i g_i) U^T S^(-1/2)`.

The same selected subspace is transported to the world frame and used for a
Gaussian-mean fusion increment. Thus the pose proposal and Gaussian geometry
obey one observability decision instead of two unrelated confidence scalars.

## Proposition 1: bounded directional gain

If `Sigma_s` and `R` are positive semidefinite and `S` is positive definite,
then every eigenvalue of `G` lies in `[0, 1]`.

**Proof.** `G` is congruent to `Sigma_s`, hence `G >= 0`. Moreover
`I-G = S^(-1/2) R S^(-1/2) >= 0`. Therefore `0 <= G <= I`, which bounds all
its eigenvalues. This prevents amplification in whitened innovation
coordinates.

## Proposition 2: exact invariance of rejected modes

Let `z = U^T S^(-1/2) delta`. The whitened applied correction is
`U^T S^(-1/2) delta_plus = diag(p_i g_i) z`. Hence for every rejected mode
`p_i=0`, its correction is exactly zero. This is stronger than covariance
inflation or a scalar confidence weight, which can still leak updates into an
unobservable direction.

For Gaussian fusion, we use the transported translational projector
`P_w = R_wc P_t R_wc^T` and updates
`mu_plus = mu + alpha P_w (mu_candidate-mu)`. Consequently every vector in
the null space of `P_w` is invariant under the geometry update.

## Proposition 3: positive-semidefinite posterior

With Joseph propagation

`Sigma_plus = (I-A) Sigma_s (I-A)^T + A R A^T`,

`Sigma_plus` is positive semidefinite for any finite `A` whenever `Sigma_s`
and `R` are positive semidefinite, because it is a sum of two congruences of
positive-semidefinite matrices. The result therefore remains valid under rank
selection and finite-precision non-symmetry; the implementation explicitly
symmetrizes the result.

## Proposition 4: bounded map-fusion step

After symmetrization and eigenvalue clipping, the geometry projector satisfies
`0 <= P_w <= I`. For `0 <= alpha <= 1`,
`||alpha P_w d||_2 <= alpha ||d||_2`. Thus a fused Gaussian mean cannot move
farther than the ordinary scalar weighted-average step, while rejected
directions remain fixed.

## Real-time safety policy

The synchronous direct RGB-D optimizer is disabled in the default real-time
configuration because experiments on office1 showed that even gated
inter-frame corrections introduced timing-dependent drift. The default path
constructs the information subspace from sparse keyframe covariance and runs
Gaussian fusion asynchronously. Direct estimates remain an ablation and may
only be enabled with an additional cross-objective validation protocol; they
are not silently counted as successful pose improvements.
