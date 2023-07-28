using Pkg

required_packages = [
    "StatsBase",
    "ArgParse"
]

for pkg in required_packages
    if !(pkg in keys(Pkg.installed()))
        Pkg.add(pkg)
        println("Installed Package: $pkg")
    else
        println("Package already installed: $pkg")
    end
end

