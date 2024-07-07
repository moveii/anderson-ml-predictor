using SparseIR
using Random, Distributions
using Trapz
using StatsBase

include("anderson_model.jl")
using .AndersonModel


struct ModelParameters

    n::Int # number of observations
    nbath::Int # number of bath sites
    u::Vector{Float64} # coulomb interaction strength
    ε::Matrix{Float64} # bath-site energy levels
    v::Matrix{Float64} # hopping amplitude
    β::Vector{Float64} # thermodynamic beta (1/T)
    ω::AbstractVector{FermionicFreq} # Matsubara frequencies

    function ModelParameters(
        n::Int, nbath::Int,
        ε_boundary::Number, v_boundary::Number, t_lower_boundary::Number, t_upper_boundary::Number,
        ε_scale::Number, v_scale::Number, temperature_scale::Number
    )::ModelParameters
        n > 0 || throw(DomainError(n, "'n' must be positive."))
        nbath > 0 || throw(DomainError(nbath, "'nbath' must be positive."))

        energies = double_exponential_distribution(ε_scale, ε_boundary, n * nbath)
        hopping_amplitudes = double_exponential_distribution(v_scale, v_boundary, n * nbath)
        temperatures = exponential_distribution(temperature_scale, t_upper_boundary - t_lower_boundary, n) .+ t_lower_boundary

        u = ones(Float64, n)

        ε = reshape(energies, n, nbath)
        ε = hcat([sort(row, by=abs) for row in eachrow(ε)]...)' # sort the energies per row and reshape the matrix again

        v = reshape(hopping_amplitudes, n, nbath)
        β = 1.0 ./ temperatures

        # TODO should this be ε_boundary, v_boundary or something else? The doc states ωmax (was εv_boundary)
        basis = SparseIR.FiniteTempBasis(Fermionic(), 1 / t_lower_boundary, ε_boundary, nothing)
        ω = SparseIR.default_matsubara_sampling_points(basis, positive_only=true)

        return new(n, nbath, u, ε, v, β, ω)
    end

end


function exponential_distribution(scale::Number, boundary::Number, n::Int)
    scale > 0 || throw(DomainError(scale, "The scale must be positive."))
    boundary > 0 || throw(DomainError(boundary, "The boundary must be positive."))
    n > 0 || throw(DomainError(n, "'n' must be positive."))

    distribution = Exponential(scale)
    values = Float64[]

    # draw n values based on the distribution and make sure there are in the boundary 
    while length(values) < n
        nsample = Int(round((n - length(values)) * 1.2))
        batch = filter(value -> value < boundary, rand(distribution, nsample))
        append!(values, batch)
    end

    return values[1:n] # return only n numbers
end


function double_exponential_distribution(scale::Number, boundary::Number, n::Int)
    # take the exponential distribution and randomly flip the sign
    return exponential_distribution(scale, boundary, n) .* rand([-1, 1], n)
end


function get_anderson_parameters(model, model_parameters::ModelParameters; τn::Int=100)
    n = model_parameters.n
    u = model_parameters.u
    ε = model_parameters.ε
    v = model_parameters.v
    β = model_parameters.β

    parameters = if model == :exact || model == :resonant
        [AndersonParameters(u[i], ε[i, :], v[i, :]) for i in 1:n]
    elseif model == :single_pole
        [AndersonParameters(u[i], ε[i, 1], v[i, 1]) for i in 1:n]
    end

    if model == :resonant
        τ = zeros(Float64, n, τn)
        Δ = zeros(Float64, n, τn)

        for i in 1:n
            τ[i, :] = collect(LinRange(0, β[i], τn)) # initalize τ from 0 to β[i] with unit spacing
            Δ[i, :] = AndersonModel.hybridisation_tau(τ[i, :], parameters[i], β[i])
        end

        vsimple = [trapz(τ[i, :], Δ[i, :]) / β[i] for i in 1:n] # integrate over Δ(τ) from 0 to β[i] with unit spacing
        parameters = [AndersonParameters(u[i], 0.0, vsimple[i]) for i in 1:n]
    end

    return parameters
end


function calculate_self_energies(ω::AbstractVector, parameters::Vector{AndersonParameters}, β::AbstractVector)
    length(parameters) == length(β) || throw(DimensionMismatch("Parameters and β must have the same length."))

    core = AndersonCore(AndersonModel.nbath(parameters[1]))
    self_energies = zeros(ComplexF64, length(parameters), length(ω))

    Threads.@threads for i in 1:length(parameters)
        self_energies[i, :] = AndersonModel.self_energies((1, 1), ω, core, parameters[i], β[i])
    end

    return self_energies
end


function generate_exact(model_parameters::Vector{ModelParameters})
    energies = []

    for parameters in model_parameters
        anderson_parameters = get_anderson_parameters(:exact, parameters)
        push!(energies, calculate_self_energies(parameters.ω, anderson_parameters, parameters.β))
    end

    return energies
end


function generate_single_pole(model_parameters::Vector{ModelParameters})
    energies = []

    for parameters in model_parameters
        anderson_parameters = get_anderson_parameters(:single_pole, parameters)
        push!(energies, calculate_self_energies(parameters.ω, anderson_parameters, parameters.β))
    end

    return energies
end


function generate_resonant(model_parameters::Vector{ModelParameters})
    energies = []

    for parameters in model_parameters
        anderson_parameters = get_anderson_parameters(:resonant, parameters, τn=100)
        push!(energies, calculate_self_energies(parameters.ω, anderson_parameters, parameters.β))
    end

    return energies
end


function bath_site_sampling(min_nbath::Int, peak_nbath::Int, max_nbath::Int, n::Int)::Vector{Int}
    min_nbath > 0 || throw(DomainError(min_nbath, "'min_nbath' must be bigger than 0."))
    max_nbath > 0 || throw(DomainError(max_nbath, "'max_nbath' must be bigger than 0."))
    peak_nbath > 0 || throw(DomainError(peak_nbath, "'peak_nbath' must be bigger than 0."))
    min_nbath < peak_nbath <= max_nbath || throw(DomainError("'min_nbath' < 'peak_nbath' <= 'max_nbath'"))

    range = max_nbath - min_nbath

    # spread the samples evenly over the given range 
    n1 = round(n * (peak_nbath - min_nbath) / range)
    n2 = round(n * (max_nbath - peak_nbath) / range)

    # make sure we still have a total of n items (because we round)
    if n1 + n2 != n
        n1 += n - n1 - n2
    end

    x1 = exponential_distribution(0.7, peak_nbath - min_nbath, Int(n1)) .* (-1) .+ peak_nbath
    x2 = []

    # if peak and max are equal, n2 is zero and must therefore be ignored
    # note: scale = n = 0 isn't supported by the distribution, because that would be unexpected behaviour
    if n2 != 0
        x2 = exponential_distribution(1.5 * sqrt(max_nbath - peak_nbath), max_nbath - peak_nbath, Int(n2)) .+ peak_nbath
    end

    samples = round.([x1..., x2...]) # round so we get discrete values

    counts = countmap(samples)
    unique_samples = collect(keys(counts))
    probabilities = collect(values(counts)) ./ n

    distribution = Categorical(probabilities)
    samples = rand(distribution, n)

    return unique_samples[samples]
end


function generate_data()

    n = 10

    min_nbath = 1
    peak_nbath = 3
    max_nbath = 5

    ε_boundary = 5.0
    v_boundary = 5.0

    t_lower_boundary = 0.02
    t_upper_boundary = 0.5

    ε_scale = 2.0
    v_scale = 0.7
    temperature_scale = 0.35

    nbaths = countmap(bath_site_sampling(min_nbath, peak_nbath, max_nbath, n))

    println("Generating $n samples with the following bath-site configuration:")
    [println("$(nlocal)x$nbath") for (nbath, nlocal) in nbaths]

    model_parameters = [ModelParameters(nlocal, nbath, ε_boundary, v_boundary, t_lower_boundary, t_upper_boundary, ε_scale, v_scale, temperature_scale) for (nbath, nlocal) in nbaths]

    single_pole_time = @elapsed single_pole = generate_single_pole(model_parameters)
    println("Single Pole calculation completed in $(single_pole_time)s")

    open("single_pole.txt", "w") do file
        write(file, string(single_pole))
    end

    resonant_time = @elapsed resonant = generate_resonant(model_parameters)
    println("Resonant calculation completed in $(resonant_time)s")

    open("resonant.txt", "w") do file
        write(file, string(resonant))
    end

    exact_time = @elapsed exact = generate_exact(model_parameters)
    println("Exact calculation completed in $(exact_time)s")

    open("exact.txt", "w") do file
        write(file, string(exact))
    end

end


Random.seed!(1) # for comparability
println("Available number of threads: ", Threads.nthreads())
generate_data()