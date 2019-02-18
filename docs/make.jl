using Documenter, GaussianEP

push!(LOAD_PATH,"../src/")
makedocs(sitename="GaussianEP")
deploydocs(
	   branch = "gh-pages",
	   repo = "github.com/abraunst/GaussianEP.git",
	   devurl = "dev",
	   versions = ["stable" => "v^", "v#.#", devurl => devurl]
)
