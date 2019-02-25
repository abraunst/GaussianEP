struct FactorGauss <: Factor
   J::Matrix{Float64}
   h::Vector{Float64}
   β::Float64
   δβ::Float64
end

FactorGauss(J, h, β = 1.0) = FactorGauss(J, h, β, 0.0)

function update!(state::EPState, ψ::FactorGauss, a::Int, ρ::Real)
    @extract state : J h Σ μ
    @assert size(J[a]) == size(ψ.J) && size(h[a]) == size(ψ.h)
    if ψ.δβ > 0
        ψ.β = ψ.δβ * size(J,1) / (0.5*μ'J*μ-h'μ) + (1-ψ.δβ) * ψ.β
    end
    J[a][1] == ψ.J[1] && return 0.0
    J[a] .= ψ.J*ψ.β
    h[a] .= ψ.h*ψ.β
    return 1.0
end

