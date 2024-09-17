# Benjamin Czasch, Patrick Berger, 2023-2024

module AndersonModel

using Fermions
using Fermions.Propagators
using SparseIR

export AndersonCore,
    AndersonParameters,
    nbath,
    nsites,
    nspins,
    hybridisation_tau,
    self_energies,
    number_operator_expectation

"""Store constant components of an Anderson model for spin 1/2 particles with a single impurity site."""
struct AndersonCore

    # space to operate in
    nsites::Int
    nspins::Int
    fockspace::FockSpace
    quantum_numbers::NSet

    # operators of the hamiltonian; we need to store them to be able to calculate various Green's functions later
    annihilator_operator::Matrix{Operator}
    bath_operator::Matrix{Operator}
    hopping_operator::Matrix{Operator}
    coulomb_operator::Operator

    function AndersonCore(nbath::Int)::AndersonCore
        nsites = nbath + 1 # because we have one impurity
        nspins = 2 # since we have spin-1/2 particles, there are only two different spin projections
        fockspace = FockSpace(nspins * nsites)
        quantum_numbers = NSzSet(fockspace)

        annihilator_operator = reshape(Fermions.annihilators(fockspace), (nsites, nspins))
        bath_operator, hopping_operator, coulomb_operator = _get_suboperators(annihilator_operator)

        return new(nsites, nspins, fockspace, quantum_numbers, annihilator_operator, bath_operator, hopping_operator, coulomb_operator)
    end

end


"""Return the annihilator operator of the Anderson core."""
function annihilator_operator(core::AndersonCore)::Matrix{Operator}
    return core.annihilator_operator
end


"""Return the number of bath sites of the Anderson core (includes the impurity site)."""
function nbath(core::AndersonCore)::Int
    return core.nsites - 1 # -1 becaue we don't want to count the impurity site
end


"""Return the number of sites of the Anderson core (includes the impurity site)."""
function nsites(core::AndersonCore)::Int
    return core.nsites
end


"""Return the number of different spin projections of the particles in the Anderson core."""
function nspins(core::AndersonCore)::Int
    return core.nspins
end


"""
Compute all suboperators that appear in an Anderson model with a given number 
of sites, provided the annihilators of the underlying fock space. 
c[1, σ] are the anihilators of the impurity site per convention.
"""
function _get_suboperators(annihilator_operator::Matrix)
    c = annihilator_operator

    nsites, nspins = size(c)
    nbath = nsites - 1

    bath_operator = Matrix{Operator}(undef, nbath, nspins)
    hopping_operator = Matrix{Operator}(undef, nbath, nspins)

    for i in 1:nbath
        for σ in 1:nspins
            # (i+1) because c[1, σ] are the anihilators of the impurity site per convention.
            bath_operator[i, σ] = c[i+1, σ]' * c[i+1, σ]
            hopping_operator[i, σ] = c[1, σ]' * c[i+1, σ] + c[i+1, σ]' * c[1, σ]
        end
    end

    # TODO shouldn't it be c[1, 1]' * c[1, 1] * c[1, 2]' * c[1, 2] (page 2)?
    coulomb_operator = c[1, 1]' * c[1, 2]' * c[1, 2] * c[1, 1]

    return bath_operator, hopping_operator, coulomb_operator
end


struct AndersonParameters

    u::Float64 # coulomb interaction strength
    ε_imp::Float64 # impurity energy level
    ε::AbstractVector{Float64} # bath-site energy levels
    v::AbstractVector{Float64} # hopping amplitude
    β::Float64 # 1 / temperature

    function AndersonParameters(u::Float64, ε_imp::Float64, ε::AbstractVector{Float64}, v::AbstractVector{Float64}, β::Float64)::AndersonParameters
        length(v) != length(ε) && throw(DomainError("There must be as many energies ($(length(ε))) as hopping amplitudes ($(length(v))."))
        return new(u, ε_imp, ε, v, β)
    end

    function AndersonParameters(u::Float64, ε_imp::Float64, ε::Float64, v::AbstractVector{Float64}, β::Float64)::AndersonParameters
        return AndersonParameters(u, ε_imp, fill(ε, (length(v),)), v, β)
    end

    function AndersonParameters(u::Float64, ε_imp::Float64, ε::AbstractVector{Float64}, v::Float64, β::Float64)::AndersonParameters
        return AndersonParameters(u, ε_imp, ε, fill(v, (length(ε),)), β)
    end

    function AndersonParameters(u::Float64, ε_imp::Float64, ε::Float64, v::Float64, β::Float64)::AndersonParameters
        return AndersonParameters(u, ε_imp, fill(ε, (1,)), fill(v, (1,)), β)
    end

end


"""Return the number of bath sites of the corresponding Anderson model (includes the impurity site)."""
function nbath(parameters::AndersonParameters)::Int
    return length(parameters.ε)
end


"""Return the number of sites of the corresponding Anderson model (includes the impurity site)."""
function nsites(parameters::AndersonParameters)::Int
    return nbath(parameters) + 1 # +1 because we have one impurity site
end


"""Print the model parameters."""
function print(parameters::AndersonParameters)
    println("U = ", parameters.u)
    println("ε = ", parameters.ε)
    println("V = ", parameters.v)
end


"""
Compute the non interacting Hamltonian for an Anderson model
c[1, σ] are the anihilators of the impurity site per convention.
"""
function non_interacting_hamiltonian(core::AndersonCore, parameters::AndersonParameters)::Operator{Float64}
    nsites(core) != nsites(parameters) && throw(DomainError("The number of sites has to match; $(nsites(core)) (core) vs. $(nsites(parameters)) (parameters)."))

    H = Operator(core.fockspace)

    # Impurity site energy level contribution
    for σ in 1:nspins(core)
        if parameters.ε_imp != 0
            H += parameters.ε_imp * core.annihilator_operator[1, σ]' * core.annihilator_operator[1, σ]
        end
    end

    for i in 1:nbath(core)
        for σ in 1:nspins(core)

            # be aware: there is a (small) difference between 0 * operator and no operator
            # to not end up with extremly small numbers instead of 0, do it this way
            if parameters.ε[i] != 0
                H += parameters.ε[i] * core.bath_operator[i, σ]
            end

            if parameters.v[i] != 0
                H += parameters.v[i] * core.hopping_operator[i, σ]
            end

        end
    end

    return H
end


"""Return the hamilton operator of the Anderson model."""
function hamiltonian(core::AndersonCore, parameters::AndersonParameters)::Operator{Float64}
    # be aware: there is a (small) difference between 0 * operator and no operator
    # to not end up with extremly small numbers instead of 0, do it this way
    return if parameters.u != 0
        non_interacting_hamiltonian(core, parameters) + parameters.u * core.coulomb_operator
    else
        non_interacting_hamiltonian(core, parameters)
    end
end


"""Compute the non-interacting hamiltonian for the given Anderson model as required by full_tau and full_freq."""
function non_interacting_hamiltonian_eigen(core::AndersonCore, parameters::AndersonParameters)::EigenPartition{Float64}
    return Fermions.Propagators.HamiltonianEigen(non_interacting_hamiltonian(core, parameters), core.quantum_numbers, parameters.β)
end


"""Compute the full hamiltonian for the given Anderson model as required by full_tau and full_freq."""
function full_hamiltonian_eigen(core::AndersonCore, parameters::AndersonParameters)::EigenPartition{Float64}
    return Fermions.Propagators.HamiltonianEigen(hamiltonian(core, parameters), core.quantum_numbers, parameters.β)
end


"""Compute the non-interacting imaginary times Green's for the given Anderson model."""
function greens_operators(site::NTuple{2,Int64}, core::AndersonCore)::Tuple{Operator,Operator}
    return (-annihilator_operator(core)[site[1], 1], annihilator_operator(core)[site[2], 1]')
end


"""Compute the non-interacting imaginary times Green's for the given Anderson model."""
function g0_tau(site::NTuple{2,Int64}, τ::AbstractVector{Float64}, core::AndersonCore, parameters::AndersonParameters)
    return Fermions.Propagators.full_tau(greens_operators(site, core), τ, non_interacting_hamiltonian_eigen(core, parameters), parameters.β)
end


"""Compute the imaginary times Green's for the given Anderson model."""
function g_tau(site::NTuple{2,Int64}, τ::AbstractVector{Float64}, core::AndersonCore, parameters::AndersonParameters)
    return Fermions.Propagators.full_tau(greens_operators(site, core), τ, full_hamiltonian_eigen(core, parameters), parameters.β)
end


"""Compute the non-interacting Matsubara Green's function for the given Anderson model."""
function g0_freq(site::NTuple{2,Int64}, frequencies::AbstractVector{FermionicFreq}, core::AndersonCore, parameters::AndersonParameters)::AbstractVector{ComplexF64}
    return Fermions.Propagators.full_freq(greens_operators(site, core), frequencies, non_interacting_hamiltonian_eigen(core, parameters), parameters.β)
end


"""Compute the Matsubara Green's function for the given Anderson model."""
function g_freq(site::NTuple{2,Int64}, frequencies::AbstractVector{FermionicFreq}, core::AndersonCore, parameters::AndersonParameters)::AbstractVector{ComplexF64}
    return Fermions.Propagators.full_freq(greens_operators(site, core), frequencies, full_hamiltonian_eigen(core, parameters), parameters.β)
end


"""Compute the Matsubara self energies for a given Anderson model."""
function self_energies(site::NTuple{2,Int64}, frequencies::AbstractVector{FermionicFreq}, core::AndersonCore, parameters::AndersonParameters)::AbstractVector{ComplexF64}
    return 1 ./ g0_freq(site, frequencies, core, parameters) - 1 ./ g_freq(site, frequencies, core, parameters)
end

end