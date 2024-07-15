# this test is for the NODE functionality of Syver repo
using Makie
using IncompressibleNavierStokes
output = "output/test_F"
T = Float32
ArrayType = Array
Re = T(1_000)
n = 256
lims = T(0), T(1)
x = LinRange(lims..., n + 1), LinRange(lims..., n + 1)
setup = Setup(x...; Re, ArrayType);
ustart = random_field(setup, T(0));
dt = T(1e-3)
trange = (T(0), T(1e-3))
trange = (T(0), T(1))
savevery = 20
saveat = 20*dt

#@time state, outputs = solve_unsteady(;
#    setup,
#    ustart,
#    tlims = trange,
#    Δt = dt,
#    processors = (
#        ehist = realtimeplotter(;
#            setup,
#            plot = energy_history_plot,
#            nupdate = 10,
#            displayfig = false,
#        ),
#        anim = animator(; setup, path = "./vorticity.mkv", nupdate = savevery),
#        espec = realtimeplotter(; setup, plot = energy_spectrum_plot, nupdate = 10),
#        log = timelogger(; nupdate = 100),
#    ),
#);
using Lux
using Random
using CUDA
using KernelAbstractions
using IncompressibleNavierStokes
using IncompressibleNavierStokes: momentum!, apply_bc_u!, project!
include("src/closure.jl")
include("src/cnn.jl")
include("src/create_les_data.jl")
include("src/filter.jl")
#include("src/fno.jl")
include("src/groupconv.jl")
include("src/NeuralClosure.jl")
include("src/training.jl")

rng = Random.default_rng()
x = create_les_data(
    D = 2,
    Re = Re,
    rng = rng,
)
x.data