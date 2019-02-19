var documenterSearchIndex = {"docs": [

{
    "location": "#GaussianEP.BinaryPrior",
    "page": "Gaussian EP Documentation",
    "title": "GaussianEP.BinaryPrior",
    "category": "type",
    "text": "Binary Prior\n\np0(x) ∝ ρ δ(x-x0) + (1-ρ) δ(x-x_1)\n\n\n\n\n\n"
},

{
    "location": "#GaussianEP.EPState",
    "page": "Gaussian EP Documentation",
    "title": "GaussianEP.EPState",
    "category": "type",
    "text": "Instantaneous state of an expectation propagation run.\n\n\n\n\n\n"
},

{
    "location": "#GaussianEP.IntervalPrior",
    "page": "Gaussian EP Documentation",
    "title": "GaussianEP.IntervalPrior",
    "category": "type",
    "text": "Interval prior\n\nParameters: l,u\n\np_0(x) = frac1u-lmathbbIlleq xleq u\n\n\n\n\n\n"
},

{
    "location": "#GaussianEP.PosteriorPrior",
    "page": "Gaussian EP Documentation",
    "title": "GaussianEP.PosteriorPrior",
    "category": "type",
    "text": "This is a fake Prior that can be used to fix experimental moments Parameters: μ, v (variance, not std)\n\n\n\n\n\n"
},

{
    "location": "#GaussianEP.Prior",
    "page": "Gaussian EP Documentation",
    "title": "GaussianEP.Prior",
    "category": "type",
    "text": "Abstract Univariate Prior type\n\n\n\n\n\n"
},

{
    "location": "#GaussianEP.SpikeSlabPrior",
    "page": "Gaussian EP Documentation",
    "title": "GaussianEP.SpikeSlabPrior",
    "category": "type",
    "text": "Spike-and-slab prior\n\nParameters: ρ,λ\n\np_0(x)  (1-ρ) δ(x) + ρ mathcalN(x0λ^-1)\n\n\n\n\n\n"
},

{
    "location": "#GaussianEP.Term",
    "page": "Gaussian EP Documentation",
    "title": "GaussianEP.Term",
    "category": "type",
    "text": "This type represents an interaction term in the energy function of the form\n\n``β_i (\\frac12 x\'Ax + x\'y + c) + M_i \\log β_i``\n\nThe complete energy function is given by\n\n``∑_i β_i (\\frac12 x\' A_i x + x\' y_i + c_i) + M_i log β_i``\n\nas is represented by an Vector{Term}. Note that c and M are only needed for paramenter learning\n\n\n\n\n\n"
},

{
    "location": "#GaussianEP.ThetaPrior",
    "page": "Gaussian EP Documentation",
    "title": "GaussianEP.ThetaPrior",
    "category": "type",
    "text": "A θ(x) prior\n\n\n\n\n\n"
},

{
    "location": "#GaussianEP.expectation_propagation-Union{Tuple{P}, Tuple{T}, Tuple{Array{Term{T},1},Array{P,1}}, Tuple{Array{Term{T},1},Array{P,1},AbstractArray{T,2}}, Tuple{Array{Term{T},1},Array{P,1},AbstractArray{T,2},AbstractArray{T,1}}} where P<:Prior where T<:Real",
    "page": "Gaussian EP Documentation",
    "title": "GaussianEP.expectation_propagation",
    "category": "method",
    "text": "expectation_propagation(H::Vector{Term{T}}, P0::Vector{Prior}, F::AbstractMatrix{T} = zeros(0,length(P0)), d::Vector{T} = zeros(size(F,1));\n    maxiter::Int = 2000,\n    callback = (x...)->nothing,\n    # state::EPState{T} = EPState{T}(sum(size(F)), size(F)[2]),\n    damp::T = 0.9,\n    epsconv::T = 1e-6,\n    maxvar::T = 1e50,\n    minvar::T = 1e-50,\n    inverter::Function = inv) where {T <: Real, P <: Prior}\n\nEP for approximate inference of\n\nP(bfx)=frac1Z exp(-frac12bfx A bfx + bfx bfy))prod_i p_i(x_i)\n\nArguments:\n\nA::Array{Term{T}}: Gaussian Term (involving only x)\nP0::Array{Prior}: Prior terms (involving x and y)\nF::AbstractMatrix{T}: If included, the unknown becomes (bfxbfy)^T and a term delta(F bfx+bfd-bfy) is added.\n\nOptional named arguments:\n\nmaxiter::Int = 2000: maximum number of iterations\ncallback = (x...)->nothing: your own function to report progress, see ProgressReporter\nstate::EPState{T} = EPState{T}(sum(size(F)), size(F)[2]): If supplied, all internal state is updated here\ndamp::T = 0.9: damping parameter\nepsconv::T = 1e-6: convergence criterion\nmaxvar::T = 1e50: maximum variance\nminvar::T = 1e-50: minimum variance\ninverter::Function = inv: inverter method\n\nExample\n\njulia> t=Term(zeros(2,2),zeros(2),1.0)\nTerm{Float64}([0.0 0.0; 0.0 0.0], [0.0, 0.0], 0.0, 1.0, 0.0, 0)\n\njulia> P=[IntervalPrior(i...) for i in [(0,1),(0,1),(-2,2)]]\n3-element Array{IntervalPrior{Int64},1}:\n IntervalPrior{Int64}(0, 1)\n IntervalPrior{Int64}(0, 1)\n IntervalPrior{Int64}(-2, 2)\n\njulia> F=[1.0 -1.0];\n\njulia> expectation_propagation([t], P, F)\n([0.499997, 0.499997, 3.66527e-15], [0.083325, 0.083325, 0.204301], [0.489862, 0.489862, 3.66599e-15], [334.018, 334.018, 0.204341], :converged)\n\n\n\n\n\n"
},

{
    "location": "#GaussianEP.ProgressReporter",
    "page": "Gaussian EP Documentation",
    "title": "GaussianEP.ProgressReporter",
    "category": "type",
    "text": "ProgressReporter\nA function object to report on a running expectation_propagation.\n\n\n\n\n\n"
},

{
    "location": "#GaussianEP.gradient-Union{Tuple{T}, Tuple{T,Any,Any}} where T<:Prior",
    "page": "Gaussian EP Documentation",
    "title": "GaussianEP.gradient",
    "category": "method",
    "text": "gradient(p0::T, μ, σ) -> nothing\n\nupdate parameters with a single learning gradient step (learning rate is stored in p0)\n\n\n\n\n\n"
},

{
    "location": "#GaussianEP.moments-Tuple{SpikeSlabPrior,Any,Any}",
    "page": "Gaussian EP Documentation",
    "title": "GaussianEP.moments",
    "category": "method",
    "text": "p = frac1(ℓ+1)((1ρ-1) e^-frac12 (μσ)^2 (2-frac11+ℓ)sqrt1+frac1ℓ+1)\n\n\n\n\n\n"
},

{
    "location": "#GaussianEP.moments-Union{Tuple{T}, Tuple{T,Any,Any}} where T<:Prior",
    "page": "Gaussian EP Documentation",
    "title": "GaussianEP.moments",
    "category": "method",
    "text": "moments(p0::T, μ, σ) where T <:Prior -> (mean, variance)\n\ninput: ``p_0, μ, σ``\n\noutput: mean and variance of\n\n`` p(x) ∝ p_0(x) \\mathcal{N}(x;μ,σ) ``\n\n\n\n\n\n"
},

{
    "location": "#",
    "page": "Gaussian EP Documentation",
    "title": "Gaussian EP Documentation",
    "category": "page",
    "text": "    CurrentModule = GaussianEP\n    DocTestSetup = quote\n    using GaussianEP\nend    Modules = [GaussianEP]"
},

{
    "location": "#GaussianEP.expectation_propagation",
    "page": "Gaussian EP Documentation",
    "title": "GaussianEP.expectation_propagation",
    "category": "function",
    "text": "expectation_propagation(H::Vector{Term{T}}, P0::Vector{Prior}, F::AbstractMatrix{T} = zeros(0,length(P0)), d::Vector{T} = zeros(size(F,1));\n    maxiter::Int = 2000,\n    callback = (x...)->nothing,\n    # state::EPState{T} = EPState{T}(sum(size(F)), size(F)[2]),\n    damp::T = 0.9,\n    epsconv::T = 1e-6,\n    maxvar::T = 1e50,\n    minvar::T = 1e-50,\n    inverter::Function = inv) where {T <: Real, P <: Prior}\n\nEP for approximate inference of\n\nP(bfx)=frac1Z exp(-frac12bfx A bfx + bfx bfy))prod_i p_i(x_i)\n\nArguments:\n\nA::Array{Term{T}}: Gaussian Term (involving only x)\nP0::Array{Prior}: Prior terms (involving x and y)\nF::AbstractMatrix{T}: If included, the unknown becomes (bfxbfy)^T and a term delta(F bfx+bfd-bfy) is added.\n\nOptional named arguments:\n\nmaxiter::Int = 2000: maximum number of iterations\ncallback = (x...)->nothing: your own function to report progress, see ProgressReporter\nstate::EPState{T} = EPState{T}(sum(size(F)), size(F)[2]): If supplied, all internal state is updated here\ndamp::T = 0.9: damping parameter\nepsconv::T = 1e-6: convergence criterion\nmaxvar::T = 1e50: maximum variance\nminvar::T = 1e-50: minimum variance\ninverter::Function = inv: inverter method\n\nExample\n\njulia> t=Term(zeros(2,2),zeros(2),1.0)\nTerm{Float64}([0.0 0.0; 0.0 0.0], [0.0, 0.0], 0.0, 1.0, 0.0, 0)\n\njulia> P=[IntervalPrior(i...) for i in [(0,1),(0,1),(-2,2)]]\n3-element Array{IntervalPrior{Int64},1}:\n IntervalPrior{Int64}(0, 1)\n IntervalPrior{Int64}(0, 1)\n IntervalPrior{Int64}(-2, 2)\n\njulia> F=[1.0 -1.0];\n\njulia> expectation_propagation([t], P, F)\n([0.499997, 0.499997, 3.66527e-15], [0.083325, 0.083325, 0.204301], [0.489862, 0.489862, 3.66599e-15], [334.018, 334.018, 0.204341], :converged)\n\n\n\n\n\n"
},

{
    "location": "#Gaussian-EP-Documentation-1",
    "page": "Gaussian EP Documentation",
    "title": "Gaussian EP Documentation",
    "category": "section",
    "text": "    expectation_propagation"
},

]}
