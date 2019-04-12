using Documenter, GaussianEP

push!(LOAD_PATH,"../src/")
makedocs(sitename = "GaussianEP",
	 modules = [GaussianEP],
	 doctest = true)
deploydocs(
	   branch = "gh-pages",
	   repo = "github.com/abraunst/GaussianEP.git",
	   versions = ["stable" => "v^"]
	  )
