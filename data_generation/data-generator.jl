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

    basis::FiniteTempBasis

    function ModelParameters(
        n::Int, nbath::Int,
        ε_lower_boundary::Number, ε_upper_boundary::Number,
        u_lower_boundary::Number,
        t_lower_boundary::Number, t_upper_boundary::Number, temperature_scale::Number,
        v_lower_boundary::Number, v_upper_boundary::Number,
        generate_distribution_plots::Bool, file_suffix::String
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

        return new(n, nbath, ε_imp, ε, u, v, β, basis)
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

function get_exact_self_energies(model_parameters::ModelParameters, suffix::String, save_on_completion::Bool=false)::Matrix{ComplexF64}
    parameters = get_anderson_parameters(model_parameters)
    length(parameters) == length(model_parameters.β) || throw(DimensionMismatch("Parameters and β must have the same length."))

    core = AndersonCore(AndersonModel.nbath(parameters[1]))
    self_energies = Array{ComplexF64}(undef, length(parameters), length(model_parameters.ω))

    @showprogress Threads.@threads for index in eachindex(parameters)
        self_energies[index, :] = AndersonModel.self_energies((1, 1), model_parameters.ω, core, parameters[index])
    end

    if save_on_completion
        save_self_energies(self_energies, suffix)
    end

    return self_energies
end

function get_exact_g0(model_parameters::ModelParameters)::Matrix{ComplexF64}
    parameters = get_anderson_parameters(model_parameters)
    length(parameters) == length(model_parameters.β) || throw(DimensionMismatch("Parameters and β must have the same length."))

    core = AndersonCore(AndersonModel.nbath(parameters[1]))
    g0 = zeros(ComplexF64, length(parameters), length(model_parameters.basis))

    for n in eachindex(parameters)
        local_basis = SparseIR.rescale(model_parameters.basis, model_parameters.β[n])
        g0[n, :] = AndersonModel.g0_freq((1, 1), SparseIR.default_matsubara_sampling_points(local_basis), core, parameters[n])
    end

    return g0
end

function get_exact_g0_tau(model_parameters::ModelParameters)::Matrix{Float64}
    parameters = get_anderson_parameters(model_parameters)
    length(parameters) == length(model_parameters.β) || throw(DimensionMismatch("Parameters and β must have the same length."))

    core = AndersonCore(AndersonModel.nbath(parameters[1]))
    g0 = zeros(Float64, length(parameters), length(model_parameters.basis))

    for n in eachindex(parameters)
        # TODO is it okay to do this? with this we change ωmax
        local_basis = SparseIR.rescale(model_parameters.basis, model_parameters.β[n])
        g0[n, :] = AndersonModel.g0_tau((1, 1), SparseIR.default_tau_sampling_points(local_basis), core, parameters[n])
    end

    return g0
end

function save_self_energies(self_energies::Matrix{ComplexF64}, suffix::String)
    # Convert complex matrix to a format suitable for saving, e.g., split into real and imag parts
    re_part = real(self_energies)
    im_part = imag(self_energies)
    df = DataFrame(RealPart=vec(re_part), ImagPart=vec(im_part))
    CSV.write("$base_folder/exact_self_energies_$suffix.csv", df)
end

function save_number_operator_expectations(model_parameters::ModelParameters, suffix::String)
    n_expectations_values = Dict{Int,Float64}()
    anderson_parameters = get_anderson_parameters(model_parameters)

    async_lock = ReentrantLock() # lock for thread-safe operations

    @showprogress Threads.@threads for n in eachindex(anderson_parameters)
        core = AndersonCore(model_parameters.nbath)
        n_expectations = number_operator_expectation((1, 1), [model_parameters.β[n]], core, anderson_parameters[n])

        lock(async_lock) do
            n_expectations_values[n] = n_expectations
        end
    end

    # we sort the result, so there is no randomness between runs
    indices = sort(collect(keys(n_expectations_values)))
    n_expectations_values = [n_expectations_values[i] for i in indices]

    df = DataFrame(Index=indices, ExpectedNumberOperator=n_expectations_values)
    CSV.write("$base_folder/number/expected_number_operator_data_$suffix.csv", df)

    scatter(indices, n_expectations_values, xlabel="Observation", ylabel="<N>", title="<N> over observations", legend=false, ylim=(0, 1))
    hline!([0.5], linestyle=:dot, color=:red) # add a dotted red line at y = 0.5

    savefig("$base_folder/number/expected_number_operator_plot_$suffix.png")
end

function hybridisation_tau_local(parameters::ModelParameters)::Matrix{Float64}

    Δτ = zeros((parameters.n, length(parameters.basis)))

    for n in 1:parameters.n
        # TODO is it okay to do this? with this we change ωmax
        local_basis = SparseIR.rescale(parameters.basis, parameters.β[n])
        sum = zeros(length(parameters.basis))

        for l in 1:length(local_basis)
            for p in 1:parameters.nbath
                sum[l] += parameters.v[n, p] * parameters.v[n, p] * local_basis.s[l] * local_basis.v[l](parameters.ε[n, p])
            end
        end

        # TODO i guess we need to create a new TauSampling object every time?
        Δτ[n, :] = evaluate(SparseIR.TauSampling(local_basis), sum)
    end

    return Δτ
end

function g0_trafo_approach(parameters::ModelParameters)::Tuple{Matrix{ComplexF64},Matrix{ComplexF64}}
    sum = zeros((parameters.n, length(parameters.basis)))
    Δν = zeros(ComplexF64, (parameters.n, length(parameters.basis)))

    for n in 1:parameters.n
        # TODO is it okay to do this? with this we change ωmax
        local_basis = SparseIR.rescale(parameters.basis, parameters.β[n])
        sum = zeros(length(parameters.basis))

        for l in 1:length(local_basis)
            for p in 1:parameters.nbath
                sum[l] += parameters.v[n, p] * parameters.v[n, p] * local_basis.s[l] * local_basis.v[l](parameters.ε[n, p])
            end
        end

        Δν[n, :] = evaluate(SparseIR.MatsubaraSampling(local_basis), -sum)
    end

    # TODO somewhere is a minus missing... the manual approach is * (-1) this one without the minus
    # the manual approach and the propagator are equal, thus it should be an error here somewhere

    g0 = zeros(ComplexF64, (parameters.n, length(parameters.basis)))

    for n in 1:parameters.n

        local_basis = SparseIR.rescale(parameters.basis, parameters.β[n])
        omegas = SparseIR.default_matsubara_sampling_points(local_basis)

        for l in 1:length(parameters.basis)
            g0[n, l] = 1 / (SparseIR.valueim(omegas[l], parameters.β[n]) - Δν[n, l])
        end

    end

    return Δν, g0
end

function g0_manual_approach(parameters::ModelParameters)::Tuple{Matrix{ComplexF64},Matrix{ComplexF64}}
    Δν = zeros(ComplexF64, (parameters.n, length(parameters.basis)))

    for n in 1:parameters.n

        local_basis = SparseIR.rescale(parameters.basis, parameters.β[n])
        omegas = SparseIR.default_matsubara_sampling_points(local_basis)

        for l in 1:length(parameters.basis)
            for p in 1:parameters.nbath
                Δν[n, l] += (parameters.v[n, p] * parameters.v[n, p]) / (SparseIR.valueim(omegas[l], parameters.β[n]) - parameters.ε[n, p])
            end
        end
    end

    g0 = zeros(ComplexF64, (parameters.n, length(parameters.basis)))

    for n in 1:parameters.n

        local_basis = SparseIR.rescale(parameters.basis, parameters.β[n])
        omegas = SparseIR.default_matsubara_sampling_points(local_basis)

        for l in 1:length(parameters.basis)
            g0[n, l] = 1 / (SparseIR.valueim(omegas[l], parameters.β[n]) - Δν[n, l])
        end

    end

    return Δν, g0
end

function generate_data()
    n = 1
    nbath = 5

    ε_lower_boundary = -5.0
    ε_upper_boundary = 5.0

    u_lower_boundary = 10^-3

    t_lower_boundary = 0.0002
    t_upper_boundary = 0.005
    temperature_scale = 0.0035

    v_lower_boundary = -5.0
    v_higher_boundary = 5.0

    distribution_plots = false
    file_suffix = "1k"

    println("Generating $n samples with $nbath bath sites.")

    model_parameters = ModelParameters(
        n, nbath,
        ε_lower_boundary, ε_upper_boundary,
        u_lower_boundary,
        t_lower_boundary, t_upper_boundary, temperature_scale,
        v_lower_boundary, v_higher_boundary,
        distribution_plots, file_suffix
    )

    println("Starting with hybridisation Function")
    println("new method for calculating:")

    Δτ = hybridisation_tau_local(model_parameters)
    println(Δτ)

    println("")
    println("old method for calculating:")

    anderson_parameters = get_anderson_parameters(model_parameters)
    for (n, parameters) in enumerate(anderson_parameters)
        local_basis = SparseIR.rescale(model_parameters.basis, model_parameters.β[n])
        tau = hybridisation_tau(SparseIR.default_tau_sampling_points(local_basis), parameters)
        println(tau)
    end

    println("")
    println("Now looking at different approaches to g0")
    println("")

    Δν, g0 = g0_trafo_approach(model_parameters)

    println("hybridisation function (transfo)")
    println(Δν)

    println("")
    println("g0 (transfo)")
    println(g0)

    Δν, g0 = g0_manual_approach(model_parameters)

    println("")
    println("hybridisation function (manual)")
    println(Δν)

    println("")
    println("g0 (manual)")
    println(g0)

    println("")
    println("g0 frequency with propagator")

    #propagator_g0 = get_exact_g0(model_parameters)
    #println(propagator_g0)

    println("")
    println("g0 tau with propagator")

    propagator_g0_tau = get_exact_g0_tau(model_parameters)
    println(propagator_g0_tau)

    println("")
    println("g0 frequency reconstruction with g0 tau")

    g0_matsubara_reconstruction = zeros(ComplexF64, (n, length(model_parameters.basis)))

    for n in 1:n
        # TODO is it okay to do this? with this we change ωmax
        local_basis = SparseIR.rescale(model_parameters.basis, model_parameters.β[n])
        gl = SparseIR.fit(SparseIR.TauSampling(local_basis), propagator_g0_tau[n, :])
        g0_matsubara_reconstruction[n, :] = SparseIR.evaluate(SparseIR.MatsubaraSampling(local_basis), gl)
    end

    println(g0_matsubara_reconstruction)

    #time = @elapsed self_ernergies = get_exact_self_energies(model_parameters, file_suffix, false)
    #println(self_ernergies)
    #println(time)

    #

    #save_number_operator_expectations(model_parameters, file_suffix)

end

Random.seed!(1) # for comparability
println("Available number of threads: ", Threads.nthreads())
generate_data()
