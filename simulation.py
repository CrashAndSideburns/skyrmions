from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np


class Simulation:
    def __init__(
        self,
        N,
        fixed_point_method,
        energy_gradient_threshold=1e-5,
        Δt_threshold=1e-5,
    ):
        # Invariant: N must be odd.
        assert N % 2 == 1, "N must be odd."
        self.N = N

        # Construct random unimodular initial conditions. We use a normal
        # distribution because it is spherically symmetric, so using it to
        # generate the x, y, and z coordinates yields points uniformly
        # distributed on the surface of a sphere.
        m = np.random.standard_normal(size=(3, N, N))
        norm = np.linalg.norm(m, axis=0)
        self.m = m / norm

        # Initialize fields related to finding fixed points.
        self.fixed_point_method = fixed_point_method
        self.energy = [fixed_point_method.energy(self.m)]
        self.ΔE = -np.inf
        self.energy_gradient_threshold = energy_gradient_threshold

        # The timestep determines the precision of the fixed point iteration.
        # If the fixed point iteration fails to converge, the timestep may have
        # to be reduced, down to some threshold.
        self.Δt = 0.1
        self.Δt_threshold = Δt_threshold
        self.time = 0

    def run(self):
        # We run the simulation until either the energy gradient falls below
        # some threshold, in which case we say that the simulation has
        # converged, or until the timestep falls below the timestep threshold,
        # in which case we say that the simulation has failed to converge.
        while (
            np.abs(self.ΔE) / self.Δt > self.energy_gradient_threshold
            and self.Δt > self.Δt_threshold
        ):
            try:
                # We apply the fixed point method to obtain a new magnetization
                # field, and compute the change in energy from the current
                # field to the new field.
                new_m = self.fixed_point_method.fixed_point(self.m, self.Δt)
                new_energy = self.fixed_point_method.energy(new_m)
                new_ΔE = new_energy - self.energy[-1]

                if new_ΔE < 0:
                    # If the fixed point method has yielded a new magnetization
                    # field with lower energy, then set it to be our current
                    # magnetization field. Unlike in the MATLAB simulation,
                    # this is the only case in which the new energy is appended
                    # to the energy list.
                    self.time += self.Δt
                    self.m = new_m
                    self.ΔE = new_ΔE
                    self.energy.append(new_energy)
                else:
                    # If the fixed point method has yielded a new magnetization
                    # field with higher energy, then shrink the timestep and
                    # try again.
                    self.Δt /= 2
            except FailedConvergenceException:
                # A FixedPointMethod should raise an exception in the event
                # that it fails to converge, in which case we shrink the
                # timestep and try again.
                self.Δt /= 2

    def topological_degree(self):
        k = np.array(
            list(range(int((self.N + 1) / 2))) + list(range(int(-(self.N - 1) / 2), 0))
        )
        y, x = np.meshgrid(k, k)

        dx = np.zeros((3, 3, self.N, self.N), dtype="complex_")
        dx[0][0][:][:] = 1j * x
        dx[1][1][:][:] = 1j * x
        dx[2][2][:][:] = 1j * x

        dy = np.zeros((3, 3, self.N, self.N), dtype="complex_")
        dy[0][0][:][:] = 1j * y
        dy[1][1][:][:] = 1j * y
        dy[2][2][:][:] = 1j * y

        dx = np.real(
            np.fft.ifft2(np.einsum("ijkl, jkl -> ikl", dx, np.fft.fft2(self.m)))
        )
        dy = np.real(
            np.fft.ifft2(np.einsum("ijkl, jkl -> ikl", dy, np.fft.fft2(self.m)))
        )

        cross_product = np.cross(dx, dy, axis=0)

        return np.pi * np.average(np.einsum("ijk, ijk -> jk", cross_product, self.m))


class FixedPointMethod(ABC):
    @abstractmethod
    def fixed_point(self, m, Δt):
        pass

    @abstractmethod
    def energy(self, m):
        pass


class FailedConvergenceException(Exception):
    pass


class GinzburgLandau(FixedPointMethod):
    def __init__(
        self, N, α, β, κ, τ=1, θ=np.pi / 2, iteration_threshold=500, tolerance=1e-9
    ):
        # Invariant: N must be odd.
        assert N % 2 == 1, "N must be odd."

        # Store parameters relevant to the computation of fixed points and
        # energies.
        self.N = N
        self.α = α
        self.iteration_threshold = iteration_threshold
        self.tolerance = tolerance

        # Compute bifurcation point.
        λ2 = -α
        λ0 = -1 - β / 2 + np.sqrt(4 * κ**2 + β**2 / 4)
        λ = λ0 + 0.01 * λ2

        # The x and y computed here correspond to the x and y components of the
        # wave vectors associated with the 2D FFT of the magnetization field.
        # The somewhat strange order of the wave vectors is defined by np.fft2.
        k = np.array(list(range(int((N + 1) / 2))) + list(range(int(-(N - 1) / 2), 0)))
        y, x = np.meshgrid(k, k)
        y = -x / np.tan(θ) + y / (τ * np.sin(θ))
        norm_squared = x**2 + y**2

        # This defines the Laplacian operator in Fourier space.
        # TODO: Include brief derivation in a comment explaining why this is
        # the form that the Laplacian takes.
        Δ = np.zeros((3, 3, N, N))
        Δ[0][0][:][:] = Δ[1][1][:][:] = Δ[2][2][:][:] = -norm_squared

        # This defines the curl operator in Fourier space.
        # TODO: Include brief derivation in a comment explaining why this is
        # the form that the curl takes.
        curl = np.zeros((3, 3, N, N), dtype="complex_")
        curl[0][2][:][:] = 1j * y
        curl[1][2][:][:] = -1j * x
        curl[2][0][:][:] = -1j * y
        curl[2][1][:][:] = 1j * x

        # This defines the identity operator.
        # HACK: I strongly suspect that there is a nicer way to do this,
        # probably using np.identity. For now, this seems to work, so I'm happy
        # to leave it.
        self.I = np.zeros((3, 3, N, N))
        self.I[0][0][:][:] = np.ones((N, N))
        self.I[1][1][:][:] = np.ones((N, N))
        self.I[2][2][:][:] = np.ones((N, N))

        # This defines the so-called e3 operator. e3[x y z]ᵀ = [0 0 z]ᵀ.
        e3 = np.zeros((3, 3, N, N))
        e3[2][2][:][:] = np.ones((N, N))

        # Construct linear term in energy gradient.
        self.L = -Δ + 2 * κ * curl + λ * self.I + β * e3

    def fixed_point(self, m, Δt):
        previous_candidate = m
        current_candidate = 10 * np.ones((3, self.N, self.N))

        N = lambda u, v: -self.α * (u + v) * (u**2 + v**2) / 4

        iteration = 0
        error = np.max(np.abs(current_candidate - previous_candidate))

        inv = np.linalg.inv(np.einsum("ijkl -> klij", self.I + Δt * self.L / 2))

        constant_term = np.real(
            np.fft.ifft2(
                np.einsum(
                    "ijkl, jkl -> ikl",
                    np.einsum("klij, jmkl -> imkl", inv, self.I - Δt * self.L / 2),
                    np.fft.fft2(m),
                )
            )
        )

        while iteration < self.iteration_threshold and error > self.tolerance:
            current_candidate = (
                np.real(
                    np.fft.ifft2(
                        np.einsum(
                            "klij, jkl -> ikl",
                            inv,
                            np.fft.fft2(Δt * N(previous_candidate, m)),
                        )
                    )
                )
                + constant_term
            )
            error = np.max(np.abs(current_candidate - previous_candidate))
            previous_candidate = current_candidate
            iteration += 1

        if error < self.tolerance:
            return current_candidate
        else:
            raise FailedConvergenceException

    def energy(self, m):
        # The FFT must be divided by N^2 to account for the fact that the grid
        # size is a constant independent of how finely or coarsely we choose to
        # discretize the grid.
        # TODO: Include brief derivation in a comment providing more
        # mathematical justification for this choice.
        fft = np.fft.fft2(m) / self.N**2
        norm = np.linalg.norm(m, axis=0)
        # Because the operators which constitute L are defined in Fourier
        # space, they are applied to the FFT of m. That this is a valid
        # technique for computing the energy follows by Plancherel's theorem,
        # as L² norm in Fourier space is the same as the L² norm in non-Fourier
        # space, which is the quantity that we are interested in.
        return (
            np.sum(np.real(np.einsum("ijkl, jkl -> ikl", self.L, fft) * np.conj(fft)))
            / 2
            + self.α * np.average(norm**4) / 4
        )


class PenalizedFerromagnetic(FixedPointMethod):
    def __init__(
        self, N, κ, h, ε, τ=1, θ=np.pi / 2, iteration_threshold=500, tolerance=1e-9
    ):
        # Invariant: N must be odd.
        assert N % 2 == 1, "N must be odd."

        # Store parameters relevant to the computation of fixed points and
        # energies.
        self.N = N
        self.ε = ε
        self.iteration_threshold = iteration_threshold
        self.tolerance = tolerance

        # The x and y computed here correspond to the x and y components of the
        # wave vectors associated with the 2D FFT of the magnetization field.
        # The somewhat strange order of the wave vectors is defined by np.fft2.
        k = np.array(list(range(int((N + 1) / 2))) + list(range(int(-(N - 1) / 2), 0)))
        y, x = np.meshgrid(k, k)
        y = -x / np.tan(θ) + y / (τ * np.sin(θ))
        norm_squared = x**2 + y**2

        # This defines the Laplacian operator in Fourier space.
        # TODO: Include brief derivation in a comment explaining why this is
        # the form that the Laplacian takes.
        Δ = np.zeros((3, 3, N, N))
        Δ[0][0][:][:] = Δ[1][1][:][:] = Δ[2][2][:][:] = -norm_squared

        # This defines the curl operator in Fourier space.
        # TODO: Include brief derivation in a comment explaining why this is
        # the form that the curl takes.
        curl = np.zeros((3, 3, N, N), dtype="complex_")
        curl[0][2][:][:] = 1j * y
        curl[1][2][:][:] = -1j * x
        curl[2][0][:][:] = -1j * y
        curl[2][1][:][:] = 1j * x

        # This defines the identity operator.
        # HACK: I strongly suspect that there is a nicer way to do this,
        # probably using np.identity. For now, this seems to work, so I'm happy
        # to leave it.
        self.I = np.zeros((3, 3, N, N))
        self.I[0][0][:][:] = np.ones((N, N))
        self.I[1][1][:][:] = np.ones((N, N))
        self.I[2][2][:][:] = np.ones((N, N))

        # This defines the Zeeman term in the gradient flow equation. Z = [0 0 h]ᵀ.
        self.Z = np.zeros((3, N, N))
        self.Z[2][:][:] = h * np.ones((N, N))

        # Construct linear term in energy gradient.
        self.L = -Δ + 2 * κ * curl + h * self.I

    def fixed_point(self, m, Δt):
        previous_candidate = m
        current_candidate = 10 * np.ones((3, self.N, self.N))

        N = (
            lambda u, v: (u + v)
            * (
                1
                - (np.linalg.norm(u, axis=0) ** 2 + np.linalg.norm(v, axis=0) ** 2) / 2
            )
            / (2 * self.ε**2)
        )

        iteration = 0
        error = np.max(np.abs(current_candidate - previous_candidate))

        inv = np.linalg.inv(np.einsum("ijkl -> klij", self.I + Δt * self.L / 2))

        constant_term = np.real(
            np.fft.ifft2(
                np.einsum(
                    "ijkl, jkl -> ikl",
                    np.einsum("klij, jmkl -> imkl", inv, self.I - Δt * self.L / 2),
                    np.fft.fft2(m),
                )
            )
        )

        while iteration < self.iteration_threshold and error > self.tolerance:
            current_candidate = (
                np.real(
                    np.fft.ifft2(
                        np.einsum(
                            "klij, jkl -> ikl",
                            inv,
                            np.fft.fft2(Δt * N(previous_candidate, m) + Δt * self.Z),
                        )
                    )
                )
                + constant_term
            )
            error = np.max(np.abs(current_candidate - previous_candidate))
            previous_candidate = current_candidate
            iteration += 1

        if error < self.tolerance:
            return current_candidate
        else:
            raise FailedConvergenceException

    def energy(self, m):
        # The FFT must be divided by N^2 to account for the fact that the grid
        # size is a constant independent of how finely or coarsely we choose to
        # discretize the grid.
        # TODO: Include brief derivation in a comment providing more
        # mathematical justification for this choice.
        fft = np.fft.fft2(m) / self.N**2
        norm = np.linalg.norm(m, axis=0)
        # Because the operators which constitute L are defined in Fourier
        # space, they are applied to the FFT of m. That this is a valid
        # technique for computing the energy follows by Plancherel's theorem,
        # as L² norm in Fourier space is the same as the L² norm in non-Fourier
        # space, which is the quantity that we are interested in.
        return (
            np.sum(np.real(np.einsum("ijkl, jkl -> ikl", self.L, fft) * np.conj(fft)))
            / 2
            + np.average((1 - norm**2) ** 2) / (4 * self.ε**2)
            - np.average(np.einsum("ijk, ijk -> jk", self.Z, m))
        )
