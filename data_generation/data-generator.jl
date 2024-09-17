# Patrick Berger, 2024

using SparseIR
using Random, Distributions
using Trapz
using StatsBase
using Plots
using ProgressMeter
using CSV
using DataFrames

include("anderson_model.jl")
using .AndersonModel

base_folder = "data"

"""
ModelParameters

A struct representing the parameters for the Anderson impurity model.

Fields:
- n::Int: Number of observations
- nbath::Int: Number of bath sites
- ε_imp::Vector{Float64}: Impurity-site energy levels
- ε::Matrix{Float64}: Bath-site energy levels
- u::Vector{Float64}: Coulomb interaction strengths
- v::Matrix{Float64}: Hopping amplitudes
- β::Vector{Float64}: Thermodynamic beta (1/T)
- basis::FiniteTempBasis: Base basis for calculations
- basis_length::Int: Length of all bases
- bases::Vector{FiniteTempBasis}: Bases scaled to individual temperatures
"""
struct ModelParameters
    n::Int # number of observations
    nbath::Int # number of bath sites

    ε_imp::Vector{Float64} # impurity-site energy level
    ε::Matrix{Float64} # bath-site energy levels

    u::Vector{Float64} # coulomb interaction strength
    v::Matrix{Float64} # hopping amplitude
    β::Vector{Float64} # thermodynamic beta (1/T)

    basis::FiniteTempBasis # base basis
    basis_length::Int # the length of all bases
    bases::Vector{FiniteTempBasis} # basis scaled to the individual temperature

    function ModelParameters(
        n::Int, nbath::Int,
        ε_lower_boundary::Number, ε_upper_boundary::Number,
        u_lower_boundary::Number,
        t_lower_boundary::Number, t_upper_boundary::Number, temperature_scale::Number,
        v_lower_boundary::Number, v_upper_boundary::Number;
        init_bases=true, generate_distribution_plots::Bool=false, file_suffix::String=""
    )::ModelParameters
        n > 0 || throw(DomainError(n, "'n' must be positive."))
        nbath > 0 || throw(DomainError(nbath, "'nbath' must be positive."))
        ε_lower_boundary < ε_upper_boundary || throw(DomainError(ε_lower_boundary, "'ε_lower_boundary' must be smaller than 'ε_upper_boundary'."))
        t_lower_boundary < t_upper_boundary || throw(DomainError(t_lower_boundary, "'t_lower_boundary' must be smaller than 't_upper_boundary'."))
        v_lower_boundary < v_upper_boundary || throw(DomainError(v_lower_boundary, "'v_lower_boundary' must be smaller than 'v_upper_boundary'."))

        energies = uniform_distribution(ε_lower_boundary, ε_upper_boundary, n * nbath)
        hopping_amplitudes = uniform_distribution(v_lower_boundary, v_upper_boundary, n * nbath)
        temperatures = exponential_distribution(temperature_scale, t_upper_boundary - t_lower_boundary, n) .+ t_lower_boundary

        # u_upper_boundary is determined by 'ε_lower_boundary' and 'ε_upper_boundary'
        u_upper_boundary = 3 * (abs(ε_lower_boundary) + abs(ε_upper_boundary))
        u_lower_boundary < u_upper_boundary || throw(DomainError(u_lower_boundary, "'u_lower_boundary' must be smaller than 'u_upper_boundary'."))

        u_scale = u_upper_boundary / 30.0  # u_upper_boundary is 30, thus the scale is 1.0 
        u = exponential_distribution(u_scale, u_upper_boundary - u_lower_boundary, n) .+ u_lower_boundary

        ε_imp = fill(0.0, (length(u),))

        ε = reshape(energies, n, nbath)
        # sort the energies per row (highest to lowest) and reshape the matrix again
        ε = hcat([sort(row, by=abs) for row in eachrow(ε)]...)'

        v = reshape(hopping_amplitudes, n, nbath)
        β = 1.0 ./ temperatures

        if generate_distribution_plots
            save_distribution_plots(energies, hopping_amplitudes, β, u, file_suffix)
        end

        # 1 / t_lower_boundary is the highest beta, whereas ε_upper_boundary is the highest energy, which defines our ωmax
        basis = SparseIR.FiniteTempBasis(Fermionic(), 1 / t_lower_boundary, ε_upper_boundary, nothing)

        if init_bases
            bases = [SparseIR.rescale(basis, β[i]) for i in 1:n] # TODO is it okay to do this? with this we change ωmax
        else
            bases = Vector{FiniteTempBasis}(undef, n)
        end

        return new(n, nbath, ε_imp, ε, u, v, β, basis, length(basis), bases)
    end

    function ModelParameters(
        n::Int, nbath::Int,
        ε_imp::Vector{Float64}, ε::Matrix{Float64},
        u::Vector{Float64}, v::Matrix{Float64}, β::Vector{Float64},
        basis::FiniteTempBasis, basis_length::Int
    )::ModelParameters
        bases = [SparseIR.rescale(basis, β[i]) for i in 1:n]
        return new(n, nbath, ε_imp, ε, u, v, β, basis, basis_length, bases)
    end
end


"""
uniform_distribution(lower_boundary::Number, upper_boundary::Number, n::Int)

Generate n samples from a uniform distribution between lower_boundary and upper_boundary.

Returns a vector of n randomly generated numbers.
"""
function uniform_distribution(lower_boundary::Number, upper_boundary::Number, n::Int)
    return rand(Uniform(lower_boundary, upper_boundary), n)
end


"""
exponential_distribution(scale::Number, boundary::Number, n::Int)

Generate n samples from an exponential distribution with given scale, bounded by 'boundary'.

Returns a vector of n randomly generated numbers.
"""
function exponential_distribution(scale::Number, boundary::Number, n::Int)
    scale > 0 || throw(DomainError(scale, "The scale must be positive."))
    boundary > 0 || throw(DomainError(boundary, "The boundary must be positive."))
    n > 0 || throw(DomainError(n, "There must be a positive number of samples 'n'."))

    distribution = Exponential(scale)
    values = Float64[]

    while length(values) < n
        nsample = Int(round((n - length(values)) * 1.2))
        batch = filter(value -> value < boundary, rand(distribution, nsample))
        append!(values, batch)
    end

    return values[1:n]
end


"""
save_distribution_plots(energies::Vector{Float64}, hopping_amplitudes::Vector{Float64}, β::Vector{Float64}, u::Vector{Float64}, suffix::String)

Generate and save histogram plots for the distributions of energies, hopping amplitudes, β (inverse temperature), and Coulomb interaction strengths.
Also saves the data to CSV files.
"""
function save_distribution_plots(energies::Vector{Float64}, hopping_amplitudes::Vector{Float64}, β::Vector{Float64}, u::Vector{Float64}, suffix::String)
    base = "$base_folder/distributions"

    histogram(energies, bins=50, title="Energy levels ε for bath-sites", xlabel="Energy", ylabel="Count", legend=false, alpha=0.7)
    savefig("$base/plots/energy_distribution_$suffix.png")

    histogram(hopping_amplitudes, bins=50, title="Hopping Amplitudes V", xlabel="Hopping Amplitude", ylabel="Count", legend=false, alpha=0.7)
    savefig("$base/plots/hopping_amplitude_distribution_$suffix.png")

    histogram(β, bins=50, title="β (1/T)", xlabel="β", ylabel="Count", legend=false, alpha=0.7)
    savefig("$base/plots/beta_distribution_$suffix.png")

    histogram(1 ./ β, bins=50, title="Temperatures", xlabel="T", ylabel="Count", legend=false, alpha=0.7)
    savefig("$base/plots/temperature_distribution_$suffix.png")

    histogram(u, bins=50, title="Coulomb Interaction Strengths U", xlabel="Coulomb Interaction Strength", ylabel="Count", legend=false, alpha=0.7)
    savefig("$base/plots/coulomb_interaction_distribution_$suffix.png")

    CSV.write("$base/energy_distribution_$suffix.csv", DataFrame(Energies=energies))
    CSV.write("$base/hopping_amplitude_distribution_$suffix.csv", DataFrame(HoppingAmplitudes=hopping_amplitudes))
    CSV.write("$base/beta_distribution_$suffix.csv", DataFrame(Beta=β))
    CSV.write("$base/coulomb_interaction_distribution_$suffix.csv", DataFrame(CoulombInteractionStrengths=u))
end


"""
get_anderson_parameters(model_parameters::ModelParameters)

Convert ModelParameters to a vector of AndersonParameters, where each instance represents its own model.

Returns a vector of AndersonParameters.
"""
function get_anderson_parameters(model_parameters::ModelParameters)
    n = model_parameters.n
    u = model_parameters.u
    ε_imp = model_parameters.ε_imp
    ε = model_parameters.ε
    v = model_parameters.v
    β = model_parameters.β

    return [AndersonParameters(u[i], ε_imp[i], ε[i, :], v[i, :], β[i]) for i in 1:n]
end


"""
hybridisation_tau(parameters::ModelParameters)::Matrix{Float64}

Calculate the hybridization function in imaginary time (τ) representation.

Returns a matrix where each row corresponds to an observation and each column to a τ point.
"""
function hybridisation_tau(parameters::ModelParameters)::Matrix{Float64}
    Δl = zeros(Float64, (parameters.n, parameters.basis_length))

    @showprogress desc = "Δl" Threads.@threads for n in 1:parameters.n
        for l in 1:parameters.basis_length
            for p in 1:parameters.nbath
                Δl[n, l] += parameters.v[n, p] * parameters.v[n, p] * parameters.bases[n].s[l] * parameters.bases[n].v[l](parameters.ε[n, p])
            end
        end
    end

    Δτ = zeros(Float64, (parameters.n, parameters.basis_length))

    @showprogress desc = "Δτ" Threads.@threads for n in 1:parameters.n
        Δτ[n, :] = evaluate(SparseIR.TauSampling(parameters.bases[n]), Δl[n, :])
    end

    return Δτ
end


"""
g0_tau(model_parameters::ModelParameters; negative::Bool=false)::Matrix{Float64}

Calculate the non-interacting Green's function in imaginary time (τ) representation.

Returns a matrix where each row corresponds to an observation and each column to a τ point.
"""
function g0_tau(model_parameters::ModelParameters; negative::Bool=false)::Matrix{Float64}
    parameters = get_anderson_parameters(model_parameters)
    core = AndersonCore(AndersonModel.nbath(parameters[1]))

    factor = negative ? -1 : 1

    g0_tau = zeros(Float64, (model_parameters.n, model_parameters.basis_length))

    @showprogress desc = "g0_tau" Threads.@threads for n in 1:model_parameters.n
        g0_tau[n, :] = AndersonModel.g0_tau((1, 1), SparseIR.default_tau_sampling_points(model_parameters.bases[n]) * factor, core, parameters[n])
    end

    return g0_tau
end


"""
g0_freq(g0_tau::Matrix{Float64}, model_parameters::ModelParameters)::Matrix{ComplexF64}

Transform the non-interacting Green's function from τ to frequency representation.

Returns a matrix where each row corresponds to an observation and each column to a frequency point.
"""
function g0_freq(g0_tau::Matrix{Float64}, model_parameters::ModelParameters)::Matrix{ComplexF64}
    g0_freq = zeros(ComplexF64, (model_parameters.n, model_parameters.basis_length))

    @showprogress desc = "g0_freq" Threads.@threads for n in 1:model_parameters.n
        basis = model_parameters.bases[n]
        gl = SparseIR.fit(SparseIR.TauSampling(basis), g0_tau[n, :])
        g0_freq[n, :] = SparseIR.evaluate(SparseIR.MatsubaraSampling(basis), gl)
    end

    return g0_freq
end


"""
g_freq(model_parameters::ModelParameters)::Matrix{ComplexF64}

Calculate the interacting Green's function in frequency representation.

Returns a matrix where each row corresponds to an observation and each column to a frequency point.
"""
function g_freq(model_parameters::ModelParameters)::Matrix{ComplexF64}
    parameters = get_anderson_parameters(model_parameters)
    core = AndersonCore(AndersonModel.nbath(parameters[1]))

    # we calculate g_tau first, and then transform it into g_freq, because it's way faster
    g_tau = zeros(Float64, (model_parameters.n, model_parameters.basis_length))

    @showprogress desc = "g_tau" Threads.@threads for n in 1:model_parameters.n
        g_tau[n, :] = AndersonModel.g_tau((1, 1), SparseIR.default_tau_sampling_points(model_parameters.bases[n]), core, parameters[n])
    end

    g_freq = zeros(ComplexF64, (model_parameters.n, model_parameters.basis_length))

    @showprogress desc = "g_freq" Threads.@threads for n in 1:model_parameters.n
        basis = model_parameters.bases[n]
        gl = SparseIR.fit(SparseIR.TauSampling(basis), g_tau[n, :])
        g_freq[n, :] = SparseIR.evaluate(SparseIR.MatsubaraSampling(basis), gl)
    end

    return g_freq
end


"""
get_occupations(model_parameters::ModelParameters)::AbstractVector{Float64}

Calculate the occupations for each observation in the model.

Returns a vector of occupation numbers.
"""
function get_occupations(model_parameters::ModelParameters)::AbstractVector{Float64}
    parameters = get_anderson_parameters(model_parameters)
    core = AndersonCore(AndersonModel.nbath(parameters[1]))

    occupations = zeros(Float64, length(parameters))

    @showprogress desc = "<n>" Threads.@threads for n in 1:model_parameters.n
        occupations[n] = -AndersonModel.g_tau((1, 1), [model_parameters.β[n]], core, parameters[n])[1]
    end

    return occupations
end


"""
sigma_tau(model_parameters::ModelParameters, g0::Matrix{ComplexF64}, g::Matrix{ComplexF64}, hf::AbstractVector{Float64})::Matrix{Float64}

Calculate the self-energy in imaginary time (τ) representation.

Returns a matrix where each row corresponds to an observation and each column to a τ point.
"""
function sigma_tau(
    model_parameters::ModelParameters,
    g0::Matrix{ComplexF64}, g::Matrix{ComplexF64}, hf::AbstractVector{Float64}
)::Matrix{Float64}
    # we have to subtract the Hartree-Fock term from Σ(τ), otherwise it doesn't transform like a Green's function with SparseIR
    sigma_iv = (1 ./ g0 - 1 ./ g) .- hf

    sigma_tau = zeros(ComplexF64, (model_parameters.n, model_parameters.basis_length))

    @showprogress desc = "Στ" Threads.@threads for n in 1:model_parameters.n
        basis = model_parameters.bases[n]
        gl = SparseIR.fit(SparseIR.MatsubaraSampling(basis), sigma_iv[n, :])
        sigma_tau[n, :] = SparseIR.evaluate(SparseIR.TauSampling(basis), gl)
    end

    return real(sigma_tau)
end


"""
save_data_to_csv(model_parameters::ModelParameters, occupations::AbstractVector{Float64}, τ::Matrix{Float64}, Δτ::Matrix{Float64}, Στ::Matrix{Float64}, hf::AbstractVector{Float64}, so::Matrix{Float64}; suffix::String="", append::Bool=false)

Save all calculated data to a CSV file.
"""
function save_data_to_csv(
    model_parameters::ModelParameters, occupations::AbstractVector{Float64},
    τ::Matrix{Float64}, Δτ::Matrix{Float64}, Στ::Matrix{Float64},
    hf::AbstractVector{Float64}, so::Matrix{Float64};
    suffix::String="", append::Bool=false
)
    if !(size(τ) == size(Δτ) == size(Στ))
        error("All matrices must have the same dimensions.")
    end

    if !(length(model_parameters.β) == length(model_parameters.u) == size(τ, 1))
        error("All vectors must have the same length.")
    end

    ncols = size(τ, 2)

    colnames_ε = ["e_$(i)" for i in 1:model_parameters.nbath]
    colnames_v = ["v_$(i)" for i in 1:model_parameters.nbath]

    colnames_τ = ["tau_$(i)" for i in 1:ncols]
    colnames_Δτ = ["hyb_tau_$(i)" for i in 1:ncols]
    colnames_so = ["so_$(i)" for i in 1:ncols]
    colnames_Στ = ["sigma_tau_$(i)" for i in 1:ncols]

    colnames = vcat("beta", "u", "occupation", colnames_ε, colnames_v, colnames_τ, colnames_Δτ, "hf", colnames_so, colnames_Στ)

    data = hcat(model_parameters.β, model_parameters.u, occupations, model_parameters.ε, model_parameters.v, τ, Δτ, hf, so, Στ)
    df = DataFrame(data, Symbol.(colnames))

    CSV.write("$base_folder/data_$suffix.csv", df; append)
end


"""
save_occupations(occupations::AbstractVector{Float64}; suffix::String="", append::Bool=false)

Save the calculated occupations to a CSV file and generate a scatter plot.
"""
function save_occupations(occupations::AbstractVector{Float64}; suffix::String="", append::Bool=false)
    df = DataFrame(occupation=occupations)
    CSV.write("$base_folder/number/expected_number_operator_data_$suffix.csv", df; append)

    scatter(occupations, xlabel="Observation", ylabel="<N>", title="<N> over observations", legend=false, ylim=(0, 1))
    hline!([0.5], linestyle=:dot, color=:red) # add a dotted red line at y = 0.5

    savefig("$base_folder/number/expected_number_operator_plot_$suffix.png")
end


"""
generate_data(model_parameters::ModelParameters; suffix="", save::Bool=false, append::Bool=false)

Generate all data for the Anderson impurity model simulation.
This includes calculating various functions (hybridization, Green's functions, self-energy) and optionally saving the results.
"""
function generate_data(model_parameters::ModelParameters; suffix="", save::Bool=false, append::Bool=false)
    Δτ = hybridisation_tau(model_parameters)

    g0_τ = g0_tau(model_parameters)
    g0_τ_neg = g0_tau(model_parameters; negative=true)

    g0 = g0_freq(g0_τ, model_parameters)
    g = g_freq(model_parameters)

    occupations = get_occupations(model_parameters)

    hf = model_parameters.u .* occupations
    so = model_parameters.u .* model_parameters.u .* g0_τ .* g0_τ .* g0_τ_neg

    Στ = sigma_tau(model_parameters, g0, g, hf)

    save || return

    τs = zeros(Float64, (model_parameters.n, model_parameters.basis_length))

    for n in 1:model_parameters.n
        τs[n, :] = SparseIR.default_tau_sampling_points(model_parameters.bases[n])
    end

    save_data_to_csv(model_parameters, occupations, τs, Δτ, Στ, hf, so; suffix, append)
    save_occupations(occupations; suffix, append)
end


"""
get_chunk(model_parameters::ModelParameters, chunk_size::Int, index::Int)::ModelParameters

Extract a chunk of the model parameters for processing in smaller batches, reducing memory needs.

Returns a new ModelParameters object containing a subset of the original data.
"""
function get_chunk(model_parameters::ModelParameters, chunk_size::Int, index::Int)::ModelParameters
    model_parameters.n % chunk_size == 0 || throw(DomainError(model_parameters.n, "'n' is not divisable by 'chunk_size'."))

    start_idx = (index - 1) * chunk_size + 1
    end_idx = index * chunk_size

    return ModelParameters(
        chunk_size,
        model_parameters.nbath,
        model_parameters.ε_imp[start_idx:end_idx],
        model_parameters.ε[start_idx:end_idx, :],
        model_parameters.u[start_idx:end_idx],
        model_parameters.v[start_idx:end_idx, :],
        model_parameters.β[start_idx:end_idx],
        model_parameters.basis,
        model_parameters.basis_length
    )
end


"""
main()

The main function that sets up and runs the Anderson impurity model simulation.
It generates samples, calculates various quantities, and optionally saves the results.
"""
function main()
    n = 50000 # number of observations to generate
    nbath = 5

    # settings for the sample distributions
    ε_lower_boundary = -5.0
    ε_upper_boundary = 5.0

    u_lower_boundary = 10^-3

    t_lower_boundary = 0.0002
    t_upper_boundary = 0.005
    temperature_scale = 0.0035

    v_lower_boundary = -5.0
    v_higher_boundary = 5.0

    # wheter plots should be generated, and how the csv files should be saved
    generate_distribution_plots = true
    file_suffix = "50k"

    # use chunking for a larger number of observations to preserve memory
    use_chunking = true
    chunk_size = 1000

    println("Generating $n samples with $nbath bath sites.")

    model_parameters = ModelParameters(
        n, nbath,
        ε_lower_boundary, ε_upper_boundary,
        u_lower_boundary,
        t_lower_boundary, t_upper_boundary, temperature_scale,
        v_lower_boundary, v_higher_boundary;
        init_bases=!use_chunking, generate_distribution_plots, file_suffix
    )

    if use_chunking
        n % chunk_size == 0 || throw(DomainError(n, "'n' is not divisable by 'chunk_size'."))

        chunks = Int(n / chunk_size)
        time = 0

        for i in 1:chunks
            chunk = get_chunk(model_parameters, chunk_size, i)
            time += @elapsed generate_data(chunk; suffix=file_suffix, save=true, append=i > 1)
        end
    else
        time = @elapsed generate_data(model_parameters; suffix=file_suffix, save=true)
    end

    println("Generating $n samples with $nbath bath sites took $(time)s.")
end

Random.seed!(1) # for comparability
println("Available number of threads: ", Threads.nthreads())
main()
