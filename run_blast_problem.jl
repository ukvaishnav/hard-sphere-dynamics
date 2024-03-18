# File: run.jl
# Author: Umesh Kumar
# Description: Run file for the hard particle dynamics simulation
# Julia Version: 1.9.4 (2023-11-14)
# Plots Version: 1.39.0 (2023-11-14)
# Version: 0.0.1 (2024-02-08)

using Plots, LaTeXStrings, HDF5, Colors, Distributions, Statistics;

include("methods_v0.3.jl") # Also imports LinearAlgebra, Match

###########################################################################
######### Input the parameters required for simulation below ##############
###########################################################################

# Initialization related parameters
dim = 2
r = 1.0
m = 1.0
N = 100000
# N = 16
pf = pi * 0.15 # rho=0.15 # pf_max = 0.785398163 (Square) or 0.906899682 (Hexagonal)
bg_crystal = "Square" # "Square" or "Hexagonal
simbox_behaviour = "Reflective" # "Periodic" or "Reflective" or "Absorbing"
cell_size = 2*r
simboxfac = 10.0
frameboxfac = 1.1
purt = 0.9
E_expl = 5.0

# Simulation related parameters
t_max = 10000.0
# dt = 0.01
max_events = 1000
max_cols = 1000

sim_time = [0.0]
event_num = [0] # Array is used so that it can be passed by reference
col_num = [0] # Array is used so that it can be passed by reference
eventType = [0]
center = [0.0, 0.0]

bradius = []
btime = []
bprint = [1.0]

t_arr = [sim_time[1]]

# Plotting related initialization parameters
annotate_particles = false
draw_cells = false
# cg = cgrad(:thermal)
cg = cgrad(:viridis)
filename = "./Images/N=$N.png"

txt = ""

###########################################################################
##### Do not change the code below unless you know what you are doing #####
###########################################################################
# Initialization of the system

position = Array{Float64, 2}(undef, N, dim)
velocity = Array{Float64, 2}(undef, N, dim)
radii = Array{Float64, 1}(undef, N)
mass = Array{Float64, 1}(undef, N)
KE_arr = Array{Float64, 1}(undef, N)
cell_id_arr = Array{Int64, 2}(undef, N, dim)

if dim==2

    println("Initializing the system in 2D.")

    p_arr, cell_list, simbox, frame_xlim, frame_ylim = generate_configuration(N, m, r, pf)

    # p_arr[1].vel = [0.6989507156372317, -0.7809582462709801]
    # p_arr[2].vel = [0.5811941737484823, -0.9963726078735184]
    # p_arr[3].vel = [-0.9473525551811688, 0.7071541547540743]
    # p_arr[4].vel = [-0.8382062007962805, 0.8443113868751746]
    # p_arr[5].vel = [-0.8480452883624572, -0.6985498086242634]
    # p_arr[6].vel = [0.14500768238030926, 0.7556188806282216]
    # p_arr[7].vel = [-0.6913843344998976, -0.2280795138880003]
    # p_arr[8].vel = [0.5839041758144243, 0.9244533121960326]
    # p_arr[9].vel = [0.44649165315274586, -0.815397302017205]
    # p_arr[10].vel = [-0.017805204732149305, -0.06671829000382323]
    # p_arr[11].vel = [0.9739508860267982, 0.9308978429570272]
    # p_arr[12].vel = [-0.32841647089827086, 0.20044627314161678]
    # p_arr[13].vel = [-0.28195377483004824, -0.010029349333047888]
    # p_arr[14].vel = [0.2737980121410184, -0.3762510045749421]
    # p_arr[15].vel = [0.009531212272928302, -0.7357476995817802]
    # p_arr[16].vel = [-0.761748372384677, 0.8549360461735691]

    # i1 = 11
    # i2 = 10
    # i3 = 7
    # i4 = 6

    Nx = Int64(ceil(sqrt(N)))

    i1 = Nx * Int64(div(Nx, 2)) + Int64(div(Nx, 2))
    i2 = Nx * Int64(div(Nx, 2)) + Int64(div(Nx, 2)) + 1
    i3 = Nx * Int64(div(Nx, 2)) + Int64(div(Nx, 2)) + Nx
    i4 = Nx * Int64(div(Nx, 2)) + Int64(div(Nx, 2)) + Nx + 1

    center = (p_arr[i1].pos + p_arr[i2].pos + p_arr[i3].pos + p_arr[i4].pos)/4

    p_arr[i1].vel = [1.5, -1.5]
    p_arr[i2].vel = [-1, 1.1]
    p_arr[i3].vel = [-1, -1.2]
    p_arr[i4].vel = [1.2, 1]

    p_arr[i1].vel = rand(Normal(0, 1), 2)
    p_arr[i2].vel = rand(Normal(0, 1), 2)
    p_arr[i3].vel = rand(Normal(0, 1), 2)
    p_arr[i4].vel = -(p_arr[i1].vel + p_arr[i2].vel + p_arr[i3].vel)

    KE_i = 0.5*sum(p_arr[i1].m * sum(p_arr[i1].vel.^2))
    KE_i += 0.5*sum(p_arr[i2].m * sum(p_arr[i2].vel.^2))
    KE_i += 0.5*sum(p_arr[i3].m * sum(p_arr[i3].vel.^2))
    KE_i += 0.5*sum(p_arr[i4].m * sum(p_arr[i4].vel.^2))

    scale_velocity = sqrt(E_expl / KE_i)

    p_arr[i1].vel *= scale_velocity
    p_arr[i2].vel *= scale_velocity
    p_arr[i3].vel *= scale_velocity
    p_arr[i4].vel *= scale_velocity

    for i in 1:N # This is for storing data in HDF5 file

        if i!=i1 && i != i2 && i != i3 && i != i4
            p_arr[i].vel = [0.0, 0.0]
        end

        position[i,:] = p_arr[i].pos
        velocity[i,:] = p_arr[i].vel
        radii[i] = p_arr[i].r
        mass[i] = p_arr[i].m
        KE_arr[i] = 0.5*mass[i]*sum(velocity[i,:].^2)

        # println("pos$i = $(position[i,:]); vel$i = $(velocity[i,:]); t$i = $(p_arr[i].t);")

    end

else
    error("Currently we have only 2D background crystal.")

end

#--------------------------------------------
# Initialize Event calender

println("System initialization is complete. Now initializing the event calender.")
println("simbox = $simbox")
root = Event(t=-Inf) # root of the event calender
initialize_event_calendar(p_arr)

println("Event calender initialization is complete. Plotting the initial frame.")
plot_frame()

println("Computing and plotting the initial density within r.")

r_arr = 0.5:0.5:(simbox[1]/(2*simboxfac))
density_arr = zeros(length(r_arr))

for (i, r) in enumerate(r_arr)
    density_arr[i] = get_density_within(r)

end;

plot(r_arr, density_arr, xlabel="r", ylabel="Density", label="Density within r", title="Density within r", lw=2)
# vline!([r_arr[50], r_arr[100]], label="Compute region", lw=2, color=:black, ls=:dash)
savefig("./Images/N=$(N)_density_within_r.png")

rho_inf = mean(density_arr[50:length(r_arr)])

#############################################
######## Simulate the system ##########
#############################################

println("Starting simulation for $N particles with density $rho_inf.")

# txt *= "====================================\n"
# txt *= "time: $(sim_time[1]) , event_num: $(event_num[1]) , col_num: $(col_num[1])\n"
# txt *= "------------------------------------\n"
# print_tree_ordered(root)


# data_file = open("Testing.txt", "w")
# write(data_file, txt)
# close(data_file)
# exit()


while sim_time[1] < t_max # col_num[1] < max_cols

    step!()

    if div(sim_time[1], 100) == bprint[1]
        bprint[1] += 1.0
        plot_frame()

    end

    # plot_frame()

    # global txt *= "====================================\n"
    # global txt *= "time: $(sim_time[1]) , event_num: $(event_num[1]) , col_num: $(col_num[1])\n"
    # global txt *= "------------------------------------\n"
    # print_tree_ordered(root)

    print("\revent_num: $(event_num[1]) , col_num: $(col_num[1]) , t = $(sim_time[1])")

end

println("\nSimulation completed. Plotting final frame.")
plot_frame()

# println("blast_radius=$bradius")
# println("blast_times=$btime")


#--------------------------------------------
# data_file = open("Testing.txt", "w")

# global txt *= "====================================\n"
# global txt *= "time: $(sim_time[1]) , event_num: $(event_num[1]) , col_num: $(col_num[1])\n"
# print_tree_ordered(root)


# write(data_file, txt)
# close(data_file)
