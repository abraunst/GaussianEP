struct FactorGraph{T<:Real,F<:Factor}
    factors::Vector{F}
    idx::Vector{Vector{Int}}
    N::Int
    P::AbstractMatrix{T}
    d::AbstractVector{T}
end


FactorGraph(factors::Vector{F}, idx::Vector{Vector{Int}}, N::Int) where {F<:Factor} = FactorGraph(factors,idx,N,Diagonal(ones(N)),zeros(N))

FactorGraph(factors::Vector{F}, idx::Vector{Vector{Int}}, S::AbstractMatrix{T}, b::AbstractVector{T} = zeros(size(S,1))) where {T<:Real, F<:Factor} = FactorGraph(factors, idx, size(S,2), nullspace(Matrix(S)), S\b)


