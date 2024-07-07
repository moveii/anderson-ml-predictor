using SparseIR
using Random, Distributions
using Trapz
using StatsBase
using Plots

include("anderson_model.jl")
using .AndersonModel

struct ModelParameters
    n::Int # number of observations
    nbath::Int # number of bath sites

    ε_imp::Vector{Float64} # impurity-site energy level
    ε::Matrix{Float64} # bath-site energy levels

    u::Vector{Float64} # coulomb interaction strength

    v::Matrix{Float64} # hopping amplitude
    β::Vector{Float64} # thermodynamic beta (1/T)
    ω::AbstractVector{FermionicFreq} # Matsubara frequencies

    function ModelParameters(
        n::Int, nbath::Int,
        ε_lower_boundary::Number, ε_upper_boundary::Number,
        u_lower_boundary::Number,
        t_lower_boundary::Number, t_upper_boundary::Number, temperature_scale::Number,
        v_boundary::Number, v_scale::Number
    )::ModelParameters
        n > 0 || throw(DomainError(n, "'n' must be positive."))
        nbath > 0 || throw(DomainError(nbath, "'nbath' must be positive."))

        energies = uniform_distribution(ε_lower_boundary, ε_upper_boundary, n * nbath)
        hopping_amplitudes = double_exponential_distribution(v_scale, v_boundary, n * nbath)
        temperatures = exponential_distribution(temperature_scale, t_upper_boundary - t_lower_boundary, n) .+ t_lower_boundary

        # u_upper_boundary is determined by 'ε_lower_boundary' and 'ε_upper_boundary'
        u_upper_boundary = 3 * (abs(ε_lower_boundary) + abs(ε_upper_boundary))
        u_scale = u_upper_boundary / 3.0  # Scale parameter for exponential distribution
        u = exponential_distribution(u_scale, u_upper_boundary, n)

        ε_imp = u ./ 2 #fill(0.0, (length(u),)) # u ./ 1

        ε = reshape(energies, n, nbath)
        ε = hcat([sort(row, by=abs) for row in eachrow(ε)]...)' # sort the energies per row and reshape the matrix again

        v = reshape(hopping_amplitudes, n, nbath)
        β = 1.0 ./ temperatures

        basis = SparseIR.FiniteTempBasis(Fermionic(), 1 / t_lower_boundary, ε_upper_boundary, nothing)
        ω = SparseIR.default_matsubara_sampling_points(basis, positive_only=true)

        return new(n, nbath, ε_imp, ε, u, v, β, ω)
    end
end

function uniform_distribution(lower_boundary::Number, upper_boundary::Number, n::Int)
    return rand(Uniform(lower_boundary, upper_boundary), n)
end

function exponential_distribution(scale::Number, boundary::Number, n::Int)
    scale > 0 || throw(DomainError(scale, "The scale must be positive."))
    boundary > 0 || throw(DomainError(boundary, "The boundary must be positive."))
    n > 0 || throw(DomainError(n, "'n' must be positive."))

    distribution = Exponential(scale)
    values = Float64[]

    while length(values) < n
        nsample = Int(round((n - length(values)) * 1.2))
        batch = filter(value -> value < boundary, rand(distribution, nsample))
        append!(values, batch)
    end

    return values[1:n]
end

function double_exponential_distribution(scale::Number, boundary::Number, n::Int)
    return exponential_distribution(scale, boundary, n) .* rand([-1, 1], n)
end

function get_anderson_parameters(model_parameters::ModelParameters)
    n = model_parameters.n
    u = model_parameters.u
    ε_imp = model_parameters.ε_imp
    ε = model_parameters.ε
    v = model_parameters.v

    parameters = [AndersonParameters(u[i], ε_imp[i], ε[i, :], v[i, :]) for i in 1:n]

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
        anderson_parameters = get_anderson_parameters(parameters)
        push!(energies, calculate_self_energies(parameters.ω, anderson_parameters, parameters.β))
    end

    return energies
end

function generate_data()
    n = 50
    nbath = 5

    ε_lower_boundary = -5.0
    ε_upper_boundary = 5.0

    u_lower_boundary = 10^-3

    t_lower_boundary = 0.0002
    t_upper_boundary = 0.005
    temperature_scale = 0.0035

    v_boundary = 5.0
    v_scale = 2.2

    println("Generating $n samples with $nbath bath sites.")

    t_lower_boundary::Number, t_upper_boundary::Number, temperature_scale::Number,
    v_boundary::Number, v_scale::Number
    model_parameters = ModelParameters(
        n, nbath,
        ε_lower_boundary, ε_upper_boundary,
        u_lower_boundary,
        t_lower_boundary, t_upper_boundary, temperature_scale,
        v_boundary, v_scale
    )

    indices = []
    n_expectations_values = []

    anderson_parameters = get_anderson_parameters(model_parameters)
    for (index, parameters) in enumerate(anderson_parameters)
        core = AndersonCore(model_parameters.nbath)
        τ = [1e-10]  # Use a small positive value close to 0 to approximate 0^+
        n_expectations = number_operator_expectation((1, 1), τ, core, parameters, model_parameters.β[index])

        push!(indices, index)
        push!(n_expectations_values, n_expectations)

        println("Number operator expectation values for observation $index: $n_expectations (with u=$(model_parameters.u[index]), β=$(model_parameters.β[index]), ε=$(model_parameters.ε[index, :])")
    end

    plot(indices, n_expectations_values, xlabel="Index", ylabel="Expected Number Operator", title="Expected Number Operator vs Index", legend=false, ylim=(0, 1))
    hline!([0.5], linestyle=:dot, color=:red) # add a dotted red line at y = 0.5

    savefig("plots/expected_number_operator_plot.png")

    #exact_time = @elapsed exact = generate_exact(model_parameters)
    #println("Exact calculation completed in $(exact_time)s")

    #open("exact.txt", "w") do file
    #    write(file, string(exact))
    #end
end

Random.seed!(1) # for comparability
println("Available number of threads: ", Threads.nthreads())
generate_data()
