import Optim
import Roots
import QuadGK
import DifferentialEquations as DE

mutable struct ActivationFunction
    func::Function
    a::Float64
    b::Float64
    vm::Float64
    vp::Float64

    function ActivationFunction(func::Function; center::Float64=0.0, optimizer=(), optimpargs=(), kwargs...)
        a, b = 0, 0;
        if isempty(optimizer)
            a, b = FindActivationFunctionInflection(func, center; optimpargs=optimpargs)
        else
            a, b = FindActivationFunctionInflection(func, center; optimizer=optimizer, optimpargs=optimpargs)
        end
        ActivationFunction(func, a, b; kwargs...)
    end
    function ActivationFunction(func::Function, a::Float64, b::Float64; zerofindalgo=(), zerofindargs=())
        vm, vp = 0, 0
        if isempty(zerofindalgo)
            vm, vp = FindActivationFunctionLandingPoints(func, a, b; zerofindargs=zerofindargs)
        else
            vm, vp = FindActivationFunctionLandingPoints(func, a, b; zerofindalgo=zerofindalgo, zerofindargs=zerofindargs)
        end
        ActivationFunction(func, a, b, vm, vp)
    end
    ActivationFunction(func::Function, a::Float64, b::Float64, vm::Float64, vp::Float64) = new(func, a, b, vm, vp)
end

function FindActivationFunctionInflection(
        func::Function, center::Float64; 
        optimizer=Optim.Brent(),
        optimpargs=()
    )
    psi(x) = func(x) - x

    valmid = psi(center)
    exp = 1.0 
    while valmid > psi(center - eps(Float64)*2.0^exp)
        exp += 1.0
    end
    pointleft = center - eps(Float64)*2.0^exp
    exp = 1.0
    while valmid < psi(center + eps(Float64)*2.0^exp)
        exp += 1.0
    end
    pointright = center + eps(Float64)*2.0^exp

    a = Optim.optimize(x->psi(x), pointleft, center, optimizer; optimpargs...).minimizer
    b = Optim.optimize(x->-psi(x), center, pointright, optimizer; optimpargs...).minimizer

    return a, b
end

function FindActivationFunctionLandingPoints(
        func::Function, a::Float64, b::Float64; 
        zerofindalgo=Roots.A42(),
        zerofindargs=()
    )
    psi(x) = func(x) - x

    vala = psi(a);
    valb = psi(b);

    exp = 1.0 
    while valb > psi(a - eps(Float64)*2.0^exp)
        exp += 1.0
    end
    pointleft_before = a - eps(Float64)*2.0^(exp-1) 
    pointleft = a - eps(Float64)*2.0^exp
    exp = 1.0
    while vala < psi(b + eps(Float64)*2.0^exp)
        exp += 1.0
    end
    pointright_before = b + eps(Float64)*2.0^(exp-1)
    pointright = b + eps(Float64)*2.0^exp

    vm = Roots.find_zero(x->psi(x)-valb, (pointleft, pointleft_before), zerofindalgo; zerofindargs...)
    vp = Roots.find_zero(x->psi(x)-vala, (pointright_before, pointright), zerofindalgo; zerofindargs...)

    return vm, vp
end

function ComputeUnitlessLeftPeriod(
        af::ActivationFunction, u::Float64;
        integralargs=()
    )
    if u < af.a
        return NaN
    end

    psi(x) = af.func(x) - x
    fm(x) = (psi(x) - psi(af.a))/(u-x)^2

    period_fixed = (psi(af.b) - psi(af.a))*(1/(u - af.vm))
    period_tm = 0.0
    try
        period_tm = QuadGK.quadgk(fm, af.vm, af.a; integralargs...)[1]
    catch e
        if e isa QuadGK.DomainError
            return NaN
        else
            rethrow(e)
        end
    end

    return period_fixed + period_tm
end

function ComputeUnitlessRightPeriod(
    af::ActivationFunction, u::Float64;
    integralargs=()
)
    if u > af.b
        return NaN
    end

    psi(x) = af.func(x) - x
    fp(x) = (psi(af.b) - psi(x))/(x-u)^2

    period_fixed = (psi(af.b) - psi(af.a))*(1/(af.vp - u))
    period_tp = 0.0
    try
        period_tp = QuadGK.quadgk(fp, af.b, af.vp; integralargs...)[1]
    catch e
        if e isa QuadGK.DomainError
            return NaN
        else
            rethrow(e)
        end
    end

    return period_fixed + period_tp
end

function ComputeUnitlessPeriod(
        af::ActivationFunction, u::Float64; 
        integralargs=()
    )
    if u < af.a || u > af.b
        return NaN
    end

    psi(x) = af.func(x) - x
    fm(x) = (psi(x) - psi(af.a))/(u-x)^2
    fp(x) = (psi(af.b) - psi(x))/(x-u)^2

    period_fixed = (psi(af.b) - psi(af.a))*(1/(af.vp - u) + 1/(u - af.vm))
    period_tm = 0.0
    period_tp = 0.0
    try
        period_tm = QuadGK.quadgk(fm, af.vm, af.a; integralargs...)[1]
        period_tp = QuadGK.quadgk(fp, af.b, af.vp; integralargs...)[1]
    catch e
        if e isa QuadGK.DomainError
            return NaN
        else
            rethrow(e)
        end
    end

    return period_fixed + period_tm + period_tp
end


function ComputeUnitlessFICurve(
        af::ActivationFunction; 
    eval_points::Vector{Float64}=collect(range(max(af.a*(1+sign(af.a)*eps(Float64)), af.a + eps(Float64)), min(af.b*(1-sign(af.b)*eps(Float64)), af.b - eps(Float64)), length=1000)),
        integralargs=()
    )
    periods = zeros(length(eval_points))
    for (i, u) in enumerate(eval_points)
        periods[i] = ComputeUnitlessPeriod(af, u; integralargs=integralargs)
    end
    return 1 ./ periods, eval_points
end

function ComputeUnitlessPeriodSim(
        af::ActivationFunction, u::Float64; 
        regultimescale=10^-5, 
        first_spike_index=1, 
        tspan=(0, 100.0), 
        solver=DE.Rodas5P(), 
        problemargs=(), 
        callbackargs=(), 
        solverargs=()
    )
    psi(x) = af.func(x) - x
    spike_times = [Float64[]]

    function spiking_dynamics!(dx, x, p, t)
        dx[1] = x[2] - u
        dx[2] = (psi(x[2]) - x[1])/regultimescale
    end

    function count_condition(x, t, integrator)
        return x[2] - (af.a+af.b)/2
    end

    function count_affect!(integrator)
        append!(integrator.p[1], integrator.t)
    end

    start_v = (af.a+af.b)/2 # Initialize in the middle of the jump
    start_z = psi(af.a)

    condition = DE.ContinuousCallback(count_condition, count_affect!, nothing; callbackargs...)
    prob = DE.ODEProblem(spiking_dynamics!, [start_z; start_v], tspan, spike_times; callback=condition, problemargs...)
    sol = DE.solve(prob, solver; solverargs...)

    if length(spike_times[1]) <= first_spike_index
        return NaN
    end
    mean_period = (spike_times[1][end] - spike_times[1][first_spike_index])/(length(spike_times[1]) - first_spike_index)
    return mean_period
end


function ComputeUnitlessFICurveSim(
        af::ActivationFunction; 
        eval_points::Vector{Float64}=collect(range(max(af.a*(1+sign(af.a)*eps(Float64)), af.a + eps(Float64)), min(af.b*(1-sign(af.b)*eps(Float64)), af.b - eps(Float64)), length=10)), 
        tspan=(), 
        period_mul=10, 
        integralargs=(),
        kwargs...
    )
    periods = zeros(length(eval_points))
    for (i, u) in enumerate(eval_points)
        cur_tspan = ()
        if isempty(tspan)
            comp_periods = ComputeUnitlessPeriod(af, u; integralargs=integralargs)
            cur_tspan = (0, period_mul*comp_periods)
        elseif cur_tspan isa AbstractVector
            cur_tspan = tspan[i]
        else
            cur_tspan = tspan
        end
        if !isnan(cur_tspan[2])
            periods[i] = ComputeUnitlessPeriodSim(af, u; tspan=cur_tspan, kwargs...)
        else
            periods[i] = NaN
        end
    end
    return 1 ./ periods, eval_points
end



function SimulateNeuron(
    af::ActivationFunction, u::Float64;
    regultimescale=10^-5,
    tspan=(0, 100.0),
    solver=DE.Rodas5P(),
    problemargs=(),
    callbackargs=(),
    solverargs=()
)
    psi(x) = af.func(x) - x
    spike_times = [Float64[], Float64[]]

    function spiking_dynamics!(dx, x, p, t)
        dx[1] = x[2] - u
        dx[2] = (psi(x[2]) - x[1])/regultimescale
    end

    function count_condition(x, t, integrator)
        return x[2] - (af.a+af.b)/2
    end

    function count_affect_up!(integrator)
        append!(integrator.p[1], integrator.t)
    end

    function count_affect_down!(integrator)
        append!(integrator.p[2], integrator.t)
    end

    start_v = (af.a+af.vm)/2 # Initialize in the middle of the left branch

    condition = DE.ContinuousCallback(count_condition, count_affect_up!, count_affect_down!; callbackargs...)
    prob = DE.ODEProblem(spiking_dynamics!, [psi(start_v); start_v], tspan, spike_times; callback=condition, problemargs...)
    sol = DE.solve(prob, solver; solverargs...)

    return sol, spike_times
end