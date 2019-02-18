using ProgressMeter, Printf

"""
    ProgressReporter
    A function object to report on a running expectation_propagation.
"""
mutable struct ProgressReporter
    t::Int
    prog::ProgressMeter.Progress
    function ProgressReporter(X)
        println(stderr, "   it Δav                            Progress")
        return new(0, ProgressMeter.Progress(10000, ""),X)
    end
end

function (r::ProgressReporter)(av, Δav, epsconv, maxiter, H, P0)
    r.t += 1
    crit1 = min(1,log(Δav)/log(max(epsconv)))
    crit2 = r.t/maxiter
    r.prog.desc = @sprintf("%5d %.2e                  ", r.t, Δav)
    ProgressMeter.update!(r.prog, Int(floor(max(crit1,crit2)*10000)), crit1>crit2 ? :green : :red)
end

