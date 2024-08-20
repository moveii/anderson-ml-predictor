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


function uniform_distribution(lower_boundary::Number, upper_boundary::Number, n::Int)
    return rand(Uniform(lower_boundary, upper_boundary), n)
end


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


# this function returns the paramaters defined in 'ModelParameters' wrapped as 'AndersonParameters', where each instance is its own model
function get_anderson_parameters(model_parameters::ModelParameters)
    n = model_parameters.n
    u = model_parameters.u
    ε_imp = model_parameters.ε_imp
    ε = model_parameters.ε
    v = model_parameters.v
    β = model_parameters.β

    return [AndersonParameters(u[i], ε_imp[i], ε[i, :], v[i, :], β[i]) for i in 1:n]
end


function delta_l(parameters::ModelParameters)::Matrix{Float64}
    Δl = zeros(Float64, (parameters.n, parameters.basis_length))

    @showprogress desc = "Δl" Threads.@threads for n in 1:parameters.n
        for l in 1:parameters.basis_length
            for p in 1:parameters.nbath
                Δl[n, l] += parameters.v[n, p] * parameters.v[n, p] * parameters.bases[n].s[l] * parameters.bases[n].v[l](parameters.ε[n, p])
            end
        end
    end

    return Δl
end


function hybridisation_tau(Δl::Matrix{Float64}, parameters::ModelParameters)::Matrix{Float64}
    Δτ = zeros(Float64, (parameters.n, parameters.basis_length))

    @showprogress desc = "Δτ" Threads.@threads for n in 1:parameters.n
        Δτ[n, :] = evaluate(SparseIR.TauSampling(parameters.bases[n]), Δl[n, :])
    end

    return Δτ
end


function g0_freq(Δl::Matrix{Float64}, parameters::ModelParameters)::Matrix{ComplexF64}
    Δν = zeros(ComplexF64, (parameters.n, parameters.basis_length))

    Threads.@threads for n in 1:parameters.n
        Δν[n, :] = evaluate(SparseIR.MatsubaraSampling(parameters.bases[n]), -Δl[n, :])
    end

    # TODO somewhere is a minus missing... the manual approach is * (-1) this one without the minus
    # the manual approach and the propagator are equal, thus it should be an error here somewhere

    g0 = zeros(ComplexF64, (parameters.n, parameters.basis_length))

    @showprogress desc = "g0" Threads.@threads for n in 1:parameters.n
        basis = parameters.bases[n]
        omegas = SparseIR.default_matsubara_sampling_points(basis)

        for l in 1:length(basis)
            g0[n, l] = 1 / (SparseIR.valueim(omegas[l], parameters.β[n]) - Δν[n, l])
        end
    end

    return g0
end


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


function get_occupations(model_parameters::ModelParameters)::AbstractVector{Float64}
    parameters = get_anderson_parameters(model_parameters)
    core = AndersonCore(AndersonModel.nbath(parameters[1]))

    occupations = zeros(Float64, length(parameters))

    @showprogress desc = "<n>" Threads.@threads for n in 1:model_parameters.n
        occupations[n] = -AndersonModel.g_tau((1, 1), [model_parameters.β[n]], core, parameters[n])[1]
    end

    return occupations
end


function sigma_tau(
    model_parameters::ModelParameters,
    g0::Matrix{ComplexF64}, g::Matrix{ComplexF64}, occupations::AbstractVector{Float64}
)::Matrix{Float64}
    sigma_iv = (1 ./ g0 - 1 ./ g) .- (model_parameters.u .* occupations)

    sigma_tau = zeros(ComplexF64, (model_parameters.n, model_parameters.basis_length))

    @showprogress desc = "Στ" Threads.@threads for n in 1:model_parameters.n
        basis = model_parameters.bases[n]
        gl = SparseIR.fit(SparseIR.MatsubaraSampling(basis), sigma_iv[n, :])
        sigma_tau[n, :] = SparseIR.evaluate(SparseIR.TauSampling(basis), gl)
    end

    return real(sigma_tau)
end


function save_data_to_csv(
    β::AbstractVector{Float64}, u::AbstractVector{Float64},
    τ::Matrix{Float64}, Δτ::Matrix{Float64}, Στ::Matrix{Float64};
    suffix::String="", append::Bool=false
)
    if !(size(τ) == size(Δτ) == size(Στ))
        error("All matrices must have the same dimensions.")
    end

    if !(length(β) == length(u) == size(τ, 1))
        error("All vectors must have the same length.")
    end

    ncols = size(τ, 2)

    colnames_τ = ["tau_$(i)" for i in 1:ncols]
    colnames_Δτ = ["hyb_tau_$(i)" for i in 1:ncols]
    colnames_Στ = ["sigma_tau_$(i)" for i in 1:ncols]

    colnames = vcat("beta", "u", colnames_τ, colnames_Δτ, colnames_Στ)

    data = hcat(β, u, τ, Δτ, Στ)
    df = DataFrame(data, Symbol.(colnames))

    CSV.write("$base_folder/data_$suffix.csv", df; append)
end


function save_occupations(occupations::AbstractVector{Float64}; suffix::String="", append::Bool=false)
    df = DataFrame(occupation=occupations)
    CSV.write("$base_folder/number/expected_number_operator_data_$suffix.csv", df; append)

    scatter(occupations, xlabel="Observation", ylabel="<N>", title="<N> over observations", legend=false, ylim=(0, 1))
    hline!([0.5], linestyle=:dot, color=:red) # add a dotted red line at y = 0.5

    savefig("$base_folder/number/expected_number_operator_plot_$suffix.png")
end


function generate_data(model_parameters::ModelParameters; suffix="", save=false, append::Bool=false)
    Δl = delta_l(model_parameters)
    Δτ = hybridisation_tau(Δl, model_parameters)

    g0 = g0_freq(Δl, model_parameters)
    g = g_freq(model_parameters)

    occupations = get_occupations(model_parameters)

    Στ = sigma_tau(model_parameters, g0, g, occupations)

    if save
        τs = zeros(Float64, (model_parameters.n, model_parameters.basis_length))

        for n in 1:model_parameters.n
            τs[n, :] = SparseIR.default_tau_sampling_points(model_parameters.bases[n])
        end

        save_data_to_csv(model_parameters.β, model_parameters.u, τs, Δτ, Στ; suffix, append)
        save_occupations(occupations; suffix, append)
    end
end


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

function main()
    n = 10000
    nbath = 5

    ε_lower_boundary = -5.0
    ε_upper_boundary = 5.0

    u_lower_boundary = 10^-3

    t_lower_boundary = 0.0002
    t_upper_boundary = 0.005
    temperature_scale = 0.0035

    v_lower_boundary = -5.0
    v_higher_boundary = 5.0

    generate_distribution_plots = true
    file_suffix = "10k"

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
