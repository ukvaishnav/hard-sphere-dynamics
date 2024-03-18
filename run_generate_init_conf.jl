# File: run.jl
# Author: Umesh Kumar
# Description: Run file for the hard particle dynamics simulation
# Julia Version: 1.9.4 (2023-11-14)
# Plots Version: 1.39.0 (2023-11-14)
# Version: 0.0.1 (2024-02-09)

using Plots, LaTeXStrings, HDF5, Colors, Distributions, Statistics;

include("methods_v0.3.jl") # Also imports LinearAlgebra, Match

###########################################################################
######### Input the parameters required for simulation below ##############
###########################################################################

# Initialization related parameters
dim = 2
r = 1.0
m = 1.0
N = 40
# N = 16
pf = pi * 0.15 # rho=0.15 # pf_max = 0.785398163 (Square) or 0.906899682 (Hexagonal)
bg_crystal = "Square" # "Square" or "Hexagonal
simbox_behaviour = "Reflective" # "Periodic" or "Reflective" or "Absorbing"
cell_size = 2*r
simboxfac = 1.0
frameboxfac = 1.1
purt = 0.0
E_expl = 5.0

# Simulation related parameters
t_max = 1000.0
# dt = 0.01
max_events = 1000
max_cols = 20000

sim_time = [0.0]
event_num = [0] # Array is used so that it can be passed by reference
col_num = [0] # Array is used so that it can be passed by reference
eventType = [0]
center = [0.0, 0.0]

bradius = []
btime = []
bprint = [1.0]

t_arr = [sim_time[1]]
v_rms_arr = []

# Plotting related initialization parameters
annotate_particles = false
draw_cells = false
draw_box = true
plot_blast_front = false
# cg = cgrad(:thermal)
cg = cgrad(:viridis)
imgloc = "./Images/"
fname = "N=$N-ev-num=$(event_num[1])-new.png"

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

    for i in 1:N # This is for storing data in HDF5 file

        position[i,:] = p_arr[i].pos
        velocity[i,:] = p_arr[i].vel
        radii[i] = p_arr[i].r
        mass[i] = p_arr[i].m
        KE_arr[i] = 0.5*mass[i]*sum(velocity[i,:].^2)

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
fname = "./Images/N_$(N)/N_$(N)_event_num_$(event_num[1]).png"
plot_frame(fname)

# df_name = "conf_0.xyz"
# save_data(df_name)

println("Computing and plotting the initial density within r.")

r_arr = 0.5:0.5:(simbox[1]/(2*simboxfac))
density_arr = zeros(length(r_arr))

for (i, r) in enumerate(r_arr)

    density_arr[i] = get_density_within(r)

end;

plot(r_arr, density_arr, xlabel="r", ylabel="Density", label="Density within r", title="Density within r", lw=2)
# vline!([r_arr[50], r_arr[100]], label="Compute region", lw=2, color=:black, ls=:dash)
savefig("./Images/N_$(N)/N_$(N)_density_within_r.png")

rho_inf = mean(density_arr[50:length(r_arr)])

#############################################
######## Simulate the system ##########
#############################################

println("Starting simulation for $N particles with density $rho_inf.")

while col_num[1] < max_cols #sim_time[1] < t_max #

    step!()

    print("\revent_num: $(event_num[1]) , col_num: $(col_num[1]) , t = $(sim_time[1])")

end

println("\nSimulation completed. Plotting final frame.")
fname = "./Images/N_$(N)/N_$(N)_event_num_$(event_num[1]).png"
plot_frame(fname)

df_name = "N_$(N)_randomized_conf.xyz"
save_data(df_name)
