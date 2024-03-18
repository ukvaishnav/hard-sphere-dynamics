# File: methods.jl.jl
# Author: Umesh Kumar
# Description: Contains all methods required for simulating the hard core interactions between the particles.
# Julia Version: 1.9.4 (2023-11-14)
# Plots Version: 1.39.0 (2023-11-14)
# Version: 0.0.1 (2024-02-01)

# This file contains all the methods required for simulating the hard core interactions between the particles.

###########################################################################
########################### Required libraries ############################
###########################################################################

using LinearAlgebra, Match;

###########################################################################
################### Particle and Event data structures ####################
###########################################################################

"""
Event data structure
    type = 0: collision,
    +/-1: +/-x cell crossing,
    +/-2: +/-y cell crossing,
    +/-3: +/-z cell crossing,
    +/-11: +/-x wall collision,
    +/-12: +/-y wall collision,
    +/-13: +/-z wall collision
"""
Base.@kwdef mutable struct Event
    t::Union{Float64, Nothing} = nothing
    type::Union{Int64, Nothing} = nothing

    p1_id::Union{Int64, Nothing} = nothing
    p2_id::Union{Int64, Nothing} = nothing

    parent::Union{Event, Nothing} = nothing
    next::Union{Event, Nothing} = nothing
    prev::Union{Event, Nothing} = nothing

end;


"""
Particle data structure
    Properties
    ----------
    m = mass
    r = radius
    dim = dimension
    pos = position
    vel = velocity
    id = particle id
    cell_id = cell id
    n_cols = number of collisions particle has undergone
    t = last time particle was updated
    cc_t = cell crossing time
    cc_type = cell crossing type (1: +x, -1: -x, 2: +y, -2: -y, 3: +z, -3: -z)
    next = next particle in neighbour list
    prev = previous particle in neighbour list
"""
Base.@kwdef mutable struct Particle
    id::Int64 = 0
    m::Float64
    r::Float64

    pos::Union{Vector{Float64}, Nothing} = nothing
    vel::Union{Vector{Float64}, Nothing} = nothing

    cell_id::Union{Vector{Int64}, Nothing} = nothing

    n_cols::Int64 = 0
    t::Float64 = 0.0 # last time particle was updated

    col_event::Union{Event, Nothing} = nothing # collision event
    cc_event::Union{Event, Nothing} = nothing # cell crossing event
    wc_event::Union{Event, Nothing} = nothing # wall collision event

    next::Union{Particle, Nothing} = nothing # next particle in cell list
    prev::Union{Particle, Nothing} = nothing # previous particle in cell list
end;


# DataStructures section ends here
# ===========================================================================
#


###########################################################################
######################## Initialization Routines ##########################
###########################################################################


# System initialization routine
"""
Initializes the configuration of Mono-disperse system in d-dimensions
(with d=2,3) and returns the array of particles,
where n is number of parameters required to describe particle
e.g. mass, radius, last_update_time, number_of_collisions.
"""
function generate_configuration(N::Int64, m::Float64, r::Float64, pf::Float64)

    if purt < 0.0 || purt > 1.0
        DomainError("Perturbation should be between 0 and 1.")
    end

    simbox = Array{Float64, 1}(undef, dim)

    Nx = ceil(sqrt(N))

    p_arr = Array{Particle, 1}(undef, N)

    if bg_crystal == "Square" # Square lattice
        if pf <= pi/4
            a = sqrt(pi / pf) * r
            shift = (simboxfac-1) * Nx * a/2
            simbox = simboxfac * a * Nx * ones(dim)

            println("N=$N, a=$a, Nx=$Nx, shift=$shift, simbox=$simbox")

            frame_shift = shift - (frameboxfac-1) * Nx * a/2
            frame_xlim = (frame_shift, simboxfac * a * Nx - frame_shift)
            frame_ylim = (frame_shift, simboxfac * a * Nx - frame_shift)

            # Number of cells in each direction
            n_cells = Int64.(ceil.(simbox/cell_size))

            cell_list = Array{Union{Nothing, Particle}, 2}(nothing, n_cells[1], n_cells[2])

            for i in 1:N
                ix, iy = div(i-1, Nx), mod(i-1, Nx)

                # Initializing the particle
                p_arr[i] = Particle(m=m, r=r)

                # fixing id of particle
                p_arr[i].id = i

                # Position specification
                p_arr[i].pos = shift + a/2 .+ [ix, iy ] * a + (a/2-r) * purt * (2*rand(dim) .- 1)

                # Velocity specification
                p_arr[i].vel = (2*rand(dim) .- 1)

                # cell_id specification
                p_arr[i].cell_id = Int64.(div.(p_arr[i].pos, cell_size)) .+ 1 # julia indexing starts from 1

                # add the particle to the cell list
                if isnothing(cell_list[p_arr[i].cell_id[1], p_arr[i].cell_id[2]]) # If no particle is assigned to this cell
                    cell_list[p_arr[i].cell_id[1], p_arr[i].cell_id[2]] = p_arr[i]

                else
                    # put the particle before the first particle in the cell
                    cell_list[p_arr[i].cell_id[1], p_arr[i].cell_id[2]].prev = p_arr[i]
                    # put the particle as the next particle of the first particle in the cell
                    p_arr[i].next = cell_list[p_arr[i].cell_id[1], p_arr[i].cell_id[2]]
                    # put the particle as the first particle in the cell
                    cell_list[p_arr[i].cell_id[1], p_arr[i].cell_id[2]] = p_arr[i]

                end

            end

            return p_arr, cell_list, simbox, frame_xlim, frame_ylim

        else
            DomainError("Packing fraction should be less than pi/4 i.e. 0.7853981633974483")

        end

    elseif bg_crystal=="Hexagonal" # Hexagonal lattice
        if pf <= pi/(2*sqrt(3))
            a = sqrt(2*pi/(sqrt(3)*pf)) * r
            shift = (simboxfac-1) * Nx * a/2
            simbox = simboxfac * a * Nx * ones(dim)

            frame_shift = shift - (frameboxfac-1) * Nx * a/2
            frame_xlim = (frame_shift, simboxfac * a * Nx - frame_shift)
            frame_ylim = (frame_shift, simboxfac * a * Nx - frame_shift)

            # Number of cells in each direction
            n_cells = Int64.(ceil.(simbox/cell_size))

            cell_list = Array{Union{Nothing, Particle}, 2}(nothing, n_cells[1], n_cells[2])

            for i in 1:N
                ix, iy = mod(i-1, Nx), div(i-1, Nx)

                # Initializing the particle
                p_arr[i] = Particle(m=m, r=r)

                # fixing id of particle
                p_arr[i].id = i

                # Position specification
                p_arr[i].pos = shift + a/2 .+ [ix * a + (1-(-1)^( mod(iy, 2) )) * a/4, iy * a * sqrt(3)/2] + (a/2-r) * purt * (2*rand(dim) .- 1)

                # Venodeity specification
                p_arr[i].vel = (2*rand(dim) .- 1)

                # cell_id specification
                p_arr[i].cell_id = Int64.( div.(p_arr[i].pos, cell_size) ) .+ 1 # julia indexing starts from 1

                if isnothing(cell_list[p_arr[i].cell_id[1], p_arr[i].cell_id[2]]) # If no particle is assigned to this cell
                    cell_list[p_arr[i].cell_id[1], p_arr[i].cell_id[2]] = p_arr[i]

                else
                    # put the particle before the first particle in the cell
                    cell_list[p_arr[i].cell_id[1], p_arr[i].cell_id[2]].prev = p_arr[i]
                    # put the particle as the next particle of the first particle in the cell
                    p_arr[i].next = cell_list[p_arr[i].cell_id[1], p_arr[i].cell_id[2]]
                    # put the particle as the first particle in the cell
                    cell_list[p_arr[i].cell_id[1], p_arr[i].cell_id[2]] = p_arr[i]

                end

            end

            return p_arr, cell_list, simbox, frame_xlim, frame_ylim

        else
            DomainError("Packing fraction should be less than pi/(2*sqrt(3)) i.e. 0.906899682117108")

        end

    else
        error("Currently we have only two implementations of 2D background crystal, Square and Hexagonal.")

    end

end; # function generate_configuration ends here


function load_conf(fname)

    data_file = open(fname, "r")

    simbox = Array{Float64, 1}(undef, dim)
    Nx = ceil(sqrt(N))
    p_arr = Array{Particle, 1}(undef, N)

    a = sqrt(pi / pf) * r
    shift = (simboxfac-1) * Nx * a/2 + a/2
    simbox = simboxfac * a * Nx * ones(dim)

    # Number of cells in each direction
    n_cells = Int64.(ceil.(simbox/cell_size))
    cell_list = Array{Union{Nothing, Particle}, 2}(nothing, n_cells[1], n_cells[2])

    println("N=$N, a=$a, Nx=$Nx, shift=$shift, simbox=$simbox")

    frame_shift = shift - (frameboxfac-1) * Nx * a/2
    frame_xlim = (frame_shift, simboxfac * a * Nx - frame_shift)
    frame_ylim = (frame_shift, simboxfac * a * Nx - frame_shift)

    center = simbox / 2
    indices = []
    rad_dist = []

    for (i, line) in enumerate(eachline(data_file))
        if i > 2
            j = i-2
            elements = split(line)

            # Initializing the particle
            p_arr[j] = Particle(m=m, r=r)

            # fixing id of particle
            p_arr[j].id = j

            p_arr[j].pos = [shift + parse(Float64, elements[2]), shift + parse(Float64, elements[3])]
            p_arr[j].vel = [0.0, 0.0]

            p_arr[j].cell_id = Int64.(div.(p_arr[j].pos, cell_size)) .+ 1 # julia indexing starts from 1

            # add the particle to the cell list
            if isnothing(cell_list[p_arr[j].cell_id[1], p_arr[j].cell_id[2]]) # If no particle is assigned to this cell
                cell_list[p_arr[j].cell_id[1], p_arr[j].cell_id[2]] = p_arr[j]

            else
                # put the particle before the first particle in the cell
                cell_list[p_arr[j].cell_id[1], p_arr[j].cell_id[2]].prev = p_arr[j]
                # put the particle as the next particle of the first particle in the cell
                p_arr[j].next = cell_list[p_arr[j].cell_id[1], p_arr[j].cell_id[2]]
                # put the particle as the first particle in the cell
                cell_list[p_arr[j].cell_id[1], p_arr[j].cell_id[2]] = p_arr[j]

            end

            if norm(p_arr[j].pos - center) < 2 * a
                push!(indices, j)
                push!(rad_dist, norm(p_arr[j].pos - center))
            end

            position[j,:] = p_arr[j].pos
            velocity[j,:] = p_arr[j].vel
            radii[j] = p_arr[j].r
            mass[j] = p_arr[j].m
            KE_arr[j] = 0.5*mass[j]*sum(velocity[j,:].^2)

        end
    end

    close(data_file)

    if length(indices) < 4
        println("Not enough particles in the central region change radius of the central region.")
        exit()

    else
        indx = argmin(rad_dist)
        i1 = indices[indx]
        deleteat!(indices, indx)
        deleteat!(rad_dist, indx)

        indx = argmin(rad_dist)
        i2 = indices[indx]
        deleteat!(indices, indx)
        deleteat!(rad_dist, indx)

        indx = argmin(rad_dist)
        i3 = indices[indx]
        deleteat!(indices, indx)
        deleteat!(rad_dist, indx)

        indx = argmin(rad_dist)
        i4 = indices[indx]
        deleteat!(indices, indx)
        deleteat!(rad_dist, indx)

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

        velocity[i1,:] = p_arr[i1].vel
        velocity[i2,:] = p_arr[i2].vel
        velocity[i3,:] = p_arr[i3].vel
        velocity[i4,:] = p_arr[i4].vel

        KE_arr[i1] = 0.5*mass[i1]*sum(velocity[i1,:].^2)
        KE_arr[i2] = 0.5*mass[i2]*sum(velocity[i2,:].^2)
        KE_arr[i3] = 0.5*mass[i3]*sum(velocity[i3,:].^2)
        KE_arr[i4] = 0.5*mass[i4]*sum(velocity[i4,:].^2)

    end

    return p_arr, cell_list, simbox, frame_xlim, frame_ylim

end; # function generate_configuration ends here



###############################################################################
# Event Calender initialization routine

"""
Initialize Event Calender with events corresponding to particle-particle
collision, cell crossings and particle-wall collisions.
"""
function initialize_event_calendar(p_arr::Array{Particle, 1})

    for p1 in p_arr

        # global txt *= "Adding events for particle $(p1.id).\n"

        # First we add collisions with neighbours
        add_collisions!(p1)

        # Second we find and the single particle events
        find_single_particle_events!(p1)

        # global txt *="\n"

    end

end; # function initialize_event_calendar ends here


"""
Add collision event for particle p1 with all its neighbours in the cell.
"""
function add_collisions!(p1::Particle)

    cell_id = p1.cell_id # cell id of the particle
    n_cells = size(cell_list)

    t_col = Inf
    p2 = nothing

    t_col_arr = []
    p2_arr = []

    for i in -1:1
        for j in -1:1

            p2_cell_id = [cell_id[1]+i, cell_id[2]+j]

            if p2_cell_id[1] >= 1 && p2_cell_id[1] <= n_cells[1] && p2_cell_id[2] >= 1 && p2_cell_id[2] <= n_cells[2]

                p_2 = cell_list[p2_cell_id[1], p2_cell_id[2]]

                while !isnothing(p_2)
                    if p_2.id != p1.id

                        tij = collision_time(p1, p_2)

                        # global txt *= "Collision time between $(p1.id) and $(p_2.id) is $tij.\n"

                        if tij < Inf && tij > 0

                            push!(t_col_arr, tij)
                            push!(p2_arr, p_2)

                        end

                        if tij < t_col

                            p2 = p_2
                            t_col = tij

                        end
                    end
                    p_2 = p_2.next

                end
            end
        end
    end

    finding = true
    while finding

        if !isempty(p2_arr)

            indx = argmin(t_col_arr)

            # global txt *= "Checking validity of collision between $(p1.id) and $(p2_arr[indx].id).\n"

            # println("Checking collision between $(p1.id) and $(p2_arr[indx].id)")

            if collision_is_valid(p1, p2_arr[indx], t_col_arr[indx]) # this also takes care of scheduling valid events
                # if collision was valid then we add the collision to the event calender
                # and remove any previously predicted collision event of p1 and p2 if they occur after this collision
                # and then we find the new collision events for partners of p1 and p2

                # global txt *= "Collision between $(p1.id) and $(p2_arr[indx].id) is added.\n"
                finding = false

            else
                # println("Collision between $(p1.id) and $(p2_arr[indx].id) is not valid.")

                deleteat!(t_col_arr, indx)
                deleteat!(p2_arr, indx)

            end
        else
            finding = false
        end

    end

end; # function add_collisions! ends here


"""
This function finds the time of cell crossing of a particle and returns the
time and the type of cell crossing. This works regardless of the dimension of
the system.
"""
function add_cell_crossing!(p1::Particle)
    t_cc = Inf
    cc_type = nothing

    for i in 1:dim
        if p1.vel[i] > 0.0
            t_cci = (p1.cell_id[i] * cell_size - p1.pos[i]) / p1.vel[i]
            if t_cci < t_cc
                t_cc = t_cci
                cc_type = i
            end
        elseif p1.vel[i] < 0.0
            t_cci = (p1.pos[i] - (p1.cell_id[i]-1)*cell_size) / abs(p1.vel[i])
            if t_cci < t_cc
                t_cc = t_cci
                cc_type = -i
            end
        end
    end

    if !isnothing(cc_type)
        ev = Event(t=p1.t + t_cc, type=cc_type, p1_id=p1.id)
        p1.cc_event = ev # give cc event link to particle
        add_event!(ev)

        # global txt *= "Cell crossing event for p$(p1.id) is added.\n"

    end

    if t_cc<0
        println("\nWhile adding cc, t_cc: $t_cc")
        println("cc_type: $cc_type")
        println("p1.pos: $(p1.pos)")
        println("p1.vel: $(p1.vel)")
        println("p1.cell_id: $(p1.cell_id)")
        println("p1.id: $(p1.id)")
	println("p1.col_event.t: $(p1.col_event.t)")

        error("New t_cc is negative (while adding cc).")

    end

end; # function add_cell_crossing! ends here


"""
Add wall collision event for particle p1.
"""

function add_wall_event!(p1::Particle)

    t_wc = Inf
    wc_type = nothing

    for i in 1:length(p1.vel)
        if p1.vel[i] > 0.0
            t_wci = (simbox[i] - p1.pos[i] - p1.r) / p1.vel[i]
            if t_wci < t_wc
                t_wc = t_wci
                wc_type = (10+i)
            end
        elseif p1.vel[i] < 0.0
            t_wci = (p1.pos[i] - p1.r) / (-p1.vel[i])
            if t_wci < t_wc
                t_wc = t_wci
                wc_type = -(10+i)
            end
        end
    end

    if t_wc < 0
        println("\nwhile finding wall event, t_wc: $t_wc")
        println("wc_type: $wc_type")
        println("p1.pos: $(p1.pos)")
        println("p1.vel: $(p1.vel)")
        println("p1.cell_id: $(p1.cell_id)")
        println("p1.id: $(p1.id)")

        error("new t_wc is negative (while finding wall event).")

    end


    if !isnothing(wc_type)
        ev = Event(t=p1.t + t_wc, type=wc_type, p1_id=p1.id)
        p1.wc_event = ev # give event link to particle
        add_event!(ev)

        # global txt *= "Wall collision event for p$(p1.id) is added.\n"

    end

end; # function add_wall_event! ends here



# Initialization section ends here
# ===========================================================================
#


###########################################################################
########################### Event Routines ################################
###########################################################################


"""
Method to add the event to the event calender which is a sorted binary tree.
"""
function add_event!(ev::Event)

    node = root

    processing = true
    while processing # If we are still processing the event
        if ev.t < node.t # the event occurs before the node event
            if isnothing(node.prev) # calender does not have event before node event
                node.prev = ev # add the event as the previous event of the node event
                ev.parent = node # set the parent of the event as the node event
                processing = false # stop processing the event

            else # calender has event before node event
                node = node.prev # move to the previous event of the node event

            end

        else # the event occurs after the node event
            if isnothing(node.next) # calender does not have event after node event
                node.next = ev # add the event as the next event of the node event
                ev.parent = node # set the parent of the event as the node event
                processing = false # stop processing the event

            else # calender has event after node event
                node = node.next # move to the next event of the node event

            end

        end

    end

end; # function add_event ends here


"""
Remove the event from the event calender
"""
function remove_event!(ev)

    if !isnothing(ev) # if ev is an Event

        if !isnothing(ev.next) # if the event has next event
            if !isnothing(ev.prev) # if the event has a previous event
                # link the previous branch of the event to first event
                # in the next branch of the event
                first_in_ev_nxt = ev.next
                while !isnothing(first_in_ev_nxt.prev)
                    first_in_ev_nxt = first_in_ev_nxt.prev

                end

                first_in_ev_nxt.prev = ev.prev
                ev.prev.parent = first_in_ev_nxt
            end

            # link the next branch with the parent of the event
            ev.next.parent = ev.parent
            if ev.parent.next == ev # if the event is the next event of its parent
                ev.parent.next = ev.next
            else # if the event is the previous event of its parent
                ev.parent.prev = ev.next
            end

        else # if the event does not have next event
            # link the previous branch of the event to the parent of the event

            if ev.parent.next == ev # if the event is the next event of its parent
                ev.parent.next = ev.prev
            else # if the event is the previous event of its parent
                ev.parent.prev = ev.prev
            end

            if !isnothing(ev.prev)
                ev.prev.parent = ev.parent
            end

        end

        # finally remove the link of the event from Event Calender
        ev.parent = nothing
        ev.next = nothing
        ev.prev = nothing

        if ev.type == 0
            p_arr[ev.p1_id].col_event = nothing
            p_arr[ev.p2_id].col_event = nothing

        elseif ev.type == 1 || ev.type == -1 || ev.type == 2 || ev.type == -2 || ev.type == 3 || ev.type == -3
            p_arr[ev.p1_id].cc_event = nothing

        elseif ev.type == 11 || ev.type == -11 || ev.type == 12 || ev.type == -12 || ev.type == 13 || ev.type == -13
            p_arr[ev.p1_id].wc_event = nothing

        end

    end

end; # function remove_event! ends here


"""
After collision event involving p1 and notthis, we find collision event for
particle p1 with its neighbours excluding notthis.
"""
function find_next_collision_after_collision!(p1, notthis)

    cell_id = p1.cell_id # cell id of the particle
    n_cells = size(cell_list)

    t_col = Inf
    p2 = nothing

    t_col_arr = []
    p2_arr = []

    for i in -1:1
        for j in -1:1

            p2_cell_id = [cell_id[1]+i, cell_id[2]+j]

            if p2_cell_id[1] >= 1 && p2_cell_id[1] <= n_cells[1] && p2_cell_id[2] >= 1 && p2_cell_id[2] <= n_cells[2]

                p_2 = cell_list[p2_cell_id[1], p2_cell_id[2]]

                while !isnothing(p_2)

                    if p_2.id != p1.id && p_2.id != notthis.id
                        tij = collision_time_after_p1(p1, p_2)

                        if tij < Inf && tij > 0
                            push!(t_col_arr, tij)
                            push!(p2_arr, p_2)
                        end

                        if tij < t_col
                            p2 = p_2
                            t_col = tij
                        end
                    end
                    p_2 = p_2.next
                end
            end
        end
    end

    finding = true
    while finding

        if !isempty(p2_arr)

            indx = argmin(t_col_arr)

            if collision_is_valid(p1, p2_arr[indx], t_col_arr[indx]) # this also takes care of scheduling valid events
                # global txt *= "Collision between p$(p1.id) and p$(p2_arr[indx].id) is added.\n"

                finding = false

            else
                # global txt *= "Collision between p$(p1.id) and p$(p2_arr[indx].id) is not valid.\n"

                deleteat!(t_col_arr, indx)
                deleteat!(p2_arr, indx)
            end
        else
            # global txt *= "No valid collision event found for p$(p1.id).\n"

            finding = false
        end

    end

end; # function find_next_collision_after_collision! ends here


"""
Checks if the collision event is valid or not. If valid then it adds the
collision event to the event calender and removes any previously predicted
collision event of p1 and p2 if they occur after this collision and then it
finds the new collision events for partners of p1 and p2.

Returns true if the collision is valid and false otherwise.
"""
function collision_is_valid(p1, p2, t_col)

    if !isnothing(p2) #likely not needed

        ev = Event(t=t_col, type=0, p1_id=p1.id, p2_id=p2.id)

        if !isnothing(p1.col_event) # if p1 already part of a collision event
            if !isnothing(p2.col_event) # if p2 already part of a collision event

                # global txt *= "Both p$(p1.id) and p$(p2.id) are part of a collision event, "

                if p1.col_event.t > ev.t && p2.col_event.t > ev.t # if the new event occurs before both

                    # global txt *= "and the collision between p$(p1.id) and p$(p2.id) occurs before their precomputed collisions.\n"

                    if p1.col_event.p1_id == p1.id
                        p1_col_part = p_arr[p1.col_event.p2_id]
                    else
                        p1_col_part = p_arr[p1.col_event.p1_id]
                    end

                    if p2.col_event.p1_id == p2.id
                        p2_col_part = p_arr[p2.col_event.p2_id]
                    else
                        p2_col_part = p_arr[p2.col_event.p1_id]
                    end

                    # global txt *= "Hence deleting the precomputed collision events involving p$(p1.id) and p$(p1_col_part.id); "
                    # global txt *= "as well collision between p$(p2.id) and p$(p2_col_part.id).\n"

                    remove_event!(p1.col_event)
                    remove_event!(p2.col_event)
                    p1.col_event = ev # give event link to particle
                    p2.col_event = ev
                    add_event!(ev) # add event to the event calender

                    find_next_collision_event_exclude_p2!(p1_col_part, p1) # replace it with add_collisions!(p1_col_part)
                    find_next_collision_event_exclude_p2!(p2_col_part, p2)

                    return true

                else

                    # global txt *= "however the collision between p$(p1.id) and p$(p2.id) does not occur before their precomputed collisions. Hence it is not a valid collision.\n"

                    return false

                end
            else # if p2 not part of a collision event

                # global txt *= "p$(p2.id) is not part of a collision event, "

                if p1.col_event.t > ev.t

                    # global txt *= "and the collision between p$(p1.id) and p$(p2.id) occurs before the precomputed collision of p$(p1.id).\n"

                    if p1.col_event.p1_id == p1.id
                        p1_col_part = p_arr[p1.col_event.p2_id]
                    else
                        p1_col_part = p_arr[p1.col_event.p1_id]
                    end

                    # global txt *= "Hence deleting the precomputed collision event involving p$(p1.id) and p$(p1_col_part.id).\n"

                    remove_event!(p1.col_event)
                    p1.col_event = ev # give event link to particle
                    p2.col_event = ev
                    add_event!(ev) # add event to the event calender

                    find_next_collision_event_exclude_p2!(p1_col_part, p1)

                    return true

                else

                    # global txt *= "however the collision between p$(p1.id) and p$(p2.id) does not occur before the precomputed collision of p$(p1.id). Hence it is not a valid collision.\n"

                    return false

                end
            end

        else # if p1 not part of a collision event

            if !isnothing(p2.col_event) # if p2 already part of a collision event

                # global txt *= "p$(p1.id) is not part of a collision event but p$(p2.id) is part of collision, "

                if p2.col_event.t > ev.t

                    # global txt *= "and the collision between p$(p1.id) and p$(p2.id) occurs before the precomputed collision of p$(p2.id).\n"

                    if p2.col_event.p1_id == p2.id
                        p2_col_part = p_arr[p2.col_event.p2_id]
                    else
                        p2_col_part = p_arr[p2.col_event.p1_id]
                    end

                    # global txt *= "Hence deleting the precomputed collision event involving p$(p2.id) and p$(p2_col_part.id).\n"

                    remove_event!(p2.col_event)
                    p1.col_event = ev # give event link to particle
                    p2.col_event = ev
                    add_event!(ev) # add event to the event calender

                    find_next_collision_event_exclude_p2!(p2_col_part, p2) ##### possible source of error

                    return true

                else

                    # global txt *= "however the collision between p$(p1.id) and p$(p2.id) does not occur before the precomputed collision of p$(p2.id). Hence it is not a valid collision.\n"

                    return false

                end
            else # if p2 not part of a collision event

                # global txt *= "Neither p$(p1.id) nor p$(p2.id) are part of a collision event, "

                p1.col_event = ev # give event link to particle
                p2.col_event = ev
                add_event!(ev) # add event to the event calender

                # global txt *= "hence the collision between p$(p1.id) and p$(p2.id) is a valid collision.\n"

                return true

            end
        end

    end
end; # function collision_is_valid ends here



"""
Find collision event for particle p1 with its neighbours excluding nothis.
"""
function find_next_collision_event_exclude_p2!(p1, notthis)

    # global txt *= "Trying to find next collision event for p$(p1.id) with its neighbours excluding p$(notthis.id).\n"

    cell_id = p1.cell_id # cell id of the particle
    n_cells = size(cell_list)

    t_col = Inf
    p2 = nothing

    t_col_arr = []
    p2_arr = []

    for i in -1:1
        for j in -1:1
            if cell_id[1]+i >= 1 && cell_id[1]+i <= n_cells[1] && cell_id[2]+j >= 1 && cell_id[2]+j <= n_cells[2]
                p_2 = cell_list[cell_id[1]+i, cell_id[2]+j]
                while !isnothing(p_2)

                    if p_2.id != p1.id && p_2.id != notthis.id
                        tij = collision_time(p1, p_2)

                        if tij < Inf && tij > 0
                            push!(t_col_arr, tij)
                            push!(p2_arr, p_2)
                        end

                        if tij < t_col
                            p2 = p_2
                            t_col = tij
                        end
                    end

                    p_2 = p_2.next

                end
            end
        end
    end

    # Check the validity of the event and add it to the event calender

    finding = true
    while finding

        if !isempty(p2_arr)

            indx = argmin(t_col_arr)

            if collision_is_valid(p1, p2_arr[indx], t_col_arr[indx]) # this also takes care of scheduling valid events
                finding = false

                # global txt *= "Collision between p$(p1.id) and p$(p2_arr[indx].id) is valid.\n"

            else
                deleteat!(t_col_arr, indx)
                deleteat!(p2_arr, indx)

            end

        else
            finding = false

        end
    end

end; # function find_next_collision_event_exclude_p2! ends here



"""
Process the collision event
"""
function process_collision(ev::Event)
    # advance particles to the time of collision
    p1, p2 = p_arr[ev.p1_id], p_arr[ev.p2_id]
    t = ev.t

    p1.pos[:] += p1.vel[:] * (t - p1.t)
    p2.pos[:] += p2.vel[:] * (t - p2.t)

    # update velocities and particle personal clocks
    p1.t = t
    p2.t = t
    p1.n_cols += 1
    p2.n_cols += 1

    update_velocities!(p1, p2)

    # remove invalidated events from the event calender
    remove_event!(p1.col_event) # remove p2.col_event is also removed
    remove_event!(p1.cc_event)
    remove_event!(p1.wc_event)

    remove_event!(p2.cc_event)
    remove_event!(p2.wc_event)

    col_num[1] += 1

end; # function process_collision ends here


function process_single_particle_event(ev::Event)
    @match ev.type begin
        1 || -1 || 2 || -2 || 3 || -3 => process_cell_crossing(ev)
        11 || -11 || 12 || -12 || 13 || -13 => process_wall_event(ev)
    end
end; # function process_single_particle_event ends here


"""
Process the cell crossing event
"""
function process_cell_crossing(ev)

    # advance particles to the time of collision
    p1 = p_arr[ev.p1_id]
    t = ev.t
    dt = t - p1.t

    # update particle's personal clock
    p1.t = t

    p1.pos[:] += p1.vel[:] * dt

    # remove the particle from the cell list
    if !isnothing(p1.prev) # if the particle has a previous particle in the cell
        p1.prev.next = p1.next
        if !isnothing(p1.next)
            p1.next.prev = p1.prev
        end

    else # if the particle is the first particle in the cell
        cell_list[p1.cell_id[1], p1.cell_id[2]] = p1.next
        if !isnothing(p1.next)
            p1.next.prev = nothing
        end

    end
    p1.next = nothing # remove the link of the particle
    p1.prev = nothing # its previous cell list

    p1.cell_id[abs(ev.type)] += sign(ev.type) # update cell_id

    # update neighbour list
    if isnothing(cell_list[p1.cell_id[1], p1.cell_id[2]]) # If no particle is assigned to this cell
        cell_list[p1.cell_id[1], p1.cell_id[2]] = p1

    else
        # put the particle before the first particle in the cell
        cell_list[p1.cell_id[1], p1.cell_id[2]].prev = p1
        # put the particle as the next particle of the first particle in the cell
        p1.next = cell_list[p1.cell_id[1], p1.cell_id[2]]
        # put the particle as the first particle in the cell
        cell_list[p1.cell_id[1], p1.cell_id[2]] = p1

    end

    # We likely do not need to delete this event yet. As it may be removed by find_next_collision_event_after_cellcrossing! function
    # if !isnothing(p1.col_event)

    #     if p1.col_event.p1_id == p1.id
    #         p1_col_part = p_arr[p1.col_event.p2_id]
    #     else
    #         p1_col_part = p_arr[p1.col_event.p1_id]
    #     end

    #     remove_event!(p1.col_event)

    #     add_collisions!(p1_col_part)

    # end

    # remove invalidated events from the event calender
    remove_event!(p1.cc_event)
    remove_event!(p1.wc_event)

end; # function process_cell_crossing ends here


"""
Process the wall event.
"""
function process_wall_event(ev)

    # advance particles to the time of collision
    p1 = p_arr[ev.p1_id]

    p1.pos = p1.pos + p1.vel * (ev.t - p1.t)
    p1.vel[abs(ev.type) - 10] *= -1

    # update particle's personal clock
    p1.t = ev.t

    if !isnothing(p1.col_event)
        if p1.col_event.p1_id == p1.id
            p1_col_part = p_arr[p1.col_event.p2_id]
        else
            p1_col_part = p_arr[p1.col_event.p1_id]
        end

        remove_event!(p1.col_event)

        add_collisions!(p1_col_part)

    end

    # remove invalidated events from the event calender
    remove_event!(p1.cc_event)
    remove_event!(p1.wc_event)

end; # function process_wall_event ends here


"""
Add single particle events for particle p1.
"""

function find_single_particle_events!(p1)

    # Second we find the cell crossing event
    add_cell_crossing!(p1)

    # Third we find wall collision event
    add_wall_event!(p1)

end; # function find_single_particle_events! ends here


"""
After collision event involving p1, we find cell crossing event for
particle p1 with its neighbours.
"""
function find_next_collision_event_after_cellcrossing!(p1)

    cell_id = p1.cell_id # cell id of the particle
    n_cells = size(cell_list)

    t_col = Inf
    p2 = nothing

    t_col_arr = []
    p2_arr = []

    for i in -1:1
        for j in -1:1
            if cell_id[1]+i >= 1 && cell_id[1]+i <= n_cells[1] && cell_id[2]+j >= 1 && cell_id[2]+j <= n_cells[2]
                p_2 = cell_list[cell_id[1]+i, cell_id[2]+j]
                while !isnothing(p_2)
                    if p_2.id != p1.id
                        tij = collision_time_after_p1(p1, p_2)

                        if tij < Inf && tij > 0
                            push!(t_col_arr, tij)
                            push!(p2_arr, p_2)
                        end

                        if tij < t_col
                            p2 = p_2
                            t_col = tij
                        end
                    end
                    p_2 = p_2.next
                end
            end
        end
    end

    if t_col < p1.t
        println("\nAfter cell crossing, t_col: $t_col")
        println("p1.pos: $(p1.pos)")
        println("p1.vel: $(p1.vel)")
        println("p1.cell_id: $(p1.cell_id)")
        println("p1.id: $(p1.id)")
        println("p1.t: $(p1.t)")
        println("p1.col_event.t: $(p1.col_event.t)")

        error("new t_col is less than p$(p1.id).t after cell crossing.")

    end

    finding = true
    while finding

        if !isempty(p2_arr)

            indx = argmin(t_col_arr)

            if collision_is_valid(p1, p2_arr[indx], t_col_arr[indx]) # this also takes care of scheduling valid events

                # global txt *= "Collision between p$(p1.id) and p$(p2_arr[indx].id) is added.\n"

                finding = false

            else

                # global txt *= "Collision between p$(p1.id) and p$(p2_arr[indx].id) is not valid.\n"

                deleteat!(t_col_arr, indx)
                deleteat!(p2_arr, indx)

            end

        else

            # global txt *= "No valid collision event found for p$(p1.id).\n"

            finding = false

        end
    end

end; # function find_next_collision_event_after_cellcrossing! ends here



###########################################################################
########################### Collision Routines ############################
###########################################################################


"""
The function returns the time of collision between two particles p1 and p2.
If the particles are not going to collide then it returns Inf.
"""
function collision_time(p1::Particle, p2::Particle)

    D = p1.r + p2.r
    t1, t2 = p1.t, p2.t
    t0 = max(t1, t2)

    pos10 = p1.pos + p1.vel * (t0 - t1)
    pos20 = p2.pos + p2.vel * (t0 - t2)

    dr = pos20 - pos10
    dv = p2.vel - p1.vel

    a = dot(dv, dv)
    b = dot(dr, dv)
    c = dot(dr, dr) - D^2

    # global txt *= "p$(p1.id) & p$(p2.id): a=$a, b=$b, c=$c\n"

    if a == 0.0
        return Inf

    else
        disc = b^2 - a*c

        if b < 0.0 && disc >= 0.0

            if (-b -sqrt(disc))/a < 0
                println("\n\n")
                println("p1.pos: $(p1.pos)")
                println("p1.vel: $(p1.vel)")
                println("p1.cell_id: $(p1.cell_id)")
                println("p1.id: $(p1.id)")
                println("p1.t: $(p1.t)")
                println("p1.n_cols: $(p1.n_cols)")

                if c < 0
                    println("Overlap occured between p$(p1.id) and p$(p2.id).")
                end

                error("Collision time is negative.")
            end

            return t0 + (-b -sqrt(disc))/a

        else
            return Inf

        end

    end

end; # function collision_time ends here


"""
Update velocity of the particles after collision
"""
function update_velocities!(p1, p2)

    r21 = p2.pos - p1.pos
    v21 = p2.vel - p1.vel
    r21_hat = r21 / norm(r21)
    v21_para = dot(v21, r21_hat) * r21_hat

    m12 = p1.m + p2.m

    p1.vel += 2*p2.m/m12 * v21_para
    p2.vel -= 2*p1.m/m12 * v21_para

end; # function update_velocities! ends here


"""
The function returns the time of collision between two particles p1 and p2.
If the particles are not going to collide then it returns Inf.

NOTE: In this function we assume that p1.t >= p2.t since p1 supposed to have
undergone collision.
"""
function collision_time_after_p1(p1::Particle, p2::Particle)
    D = p1.r + p2.r

    pos20 = p2.pos + p2.vel * (p1.t - p2.t)

    dr = pos20 - p1.pos
    dv = p2.vel - p1.vel

    a = dot(dv, dv)
    b = dot(dr, dv)
    c = dot(dr, dr) - D^2

    if c < 0
        error("Overlap between p$(p1.id) and p$(p2.id).")

    end

    if a == 0.0
        return Inf

    else
        disc = b^2 - a*c

        if b <= 0.0 && disc >= 0.0
            return p1.t +(-b - sqrt(disc))/a

        else
            return Inf

        end

    end

end; # function collision_time_after_p1 ends here



function print_tree_ordered(node::Union{Event, Nothing})
    if node === nothing
        return
    end

    print_tree_ordered(node.prev)
    # global txt *= "t=$(node.t), ev_type=$(node.type), ev.p1_id=$(node.p1_id), ev.p2_id= $(node.p2_id)\n"
    # if node.type == 0
    #     if node.p1_id==1 || node.p2_id==1
    #         global txt *= "p1.col_event.p1_id: $(p_arr[1].col_event.p1_id), p1.col_event.p1_id: $(p_arr[1].col_event.p2_id)\n"
    #     end
    # end
    print_tree_ordered(node.next)
end



###########################################################################
########################## Simulation Routine #############################
###########################################################################

"""
Advances the simulation by event at a time.
"""
function step!()

    ev = root.next

    # find the first event
    while !isnothing(ev.prev)
        ev = ev.prev
    end

    position[:] += velocity[:] * (ev.t - sim_time[1])
    KE_arr[:] = 0.5 * mass[:] .* sum(velocity.^2, dims=2)[:]

    sim_time[1] = ev.t # update the simulation time
    eventType[1] = ev.type

    # increase event count
    event_num[1] += 1

    # process the event based on its type
    @match ev.type begin
        0 => begin
                process_collision(ev)
                # global txt *= "Processed collision event between p$(ev.p1_id) and p$(ev.p2_id).\n"

                # global txt *= "pos$(ev.p1_id) = $(p_arr[ev.p1_id].pos); vel$(ev.p1_id) = $(p_arr[ev.p1_id].vel); t$(ev.p1_id) = $(p_arr[ev.p1_id].t);\n"
                # global txt *= "pos$(ev.p2_id) = $(p_arr[ev.p2_id].pos); vel$(ev.p2_id) = $(p_arr[ev.p2_id].vel); t$(ev.p2_id) = $(p_arr[ev.p2_id].t);\n"

                velocity[ev.p1_id, :] = p_arr[ev.p1_id].vel
                velocity[ev.p2_id, :] = p_arr[ev.p2_id].vel

                find_next_collision_after_collision!(p_arr[ev.p1_id], p_arr[ev.p2_id])
                find_next_collision_after_collision!(p_arr[ev.p2_id], p_arr[ev.p1_id])

                find_single_particle_events!(p_arr[ev.p1_id])
                find_single_particle_events!(p_arr[ev.p2_id])

            end
        _ => begin
                process_single_particle_event(ev)
                # global txt *= "Processed single particle event of type $(ev.type) for p$(ev.p1_id).\n"

                velocity[ev.p1_id, :] = p_arr[ev.p1_id].vel

                find_next_collision_event_after_cellcrossing!(p_arr[ev.p1_id])

                find_single_particle_events!(p_arr[ev.p1_id])

            end
    end


    ##############################################

end; # function step! ends here


###########################################################################
#################### Computing Physical Quantities ########################
###########################################################################

function rms_velocity(v_sq_arr, t_arr)

    v_rms_arr = zeros(N, size(t_arr)[1])
    v_rms_arr[:,1] = v_sq_arr[:,1]

    for i in 2:size(t_arr)[1]
        for j in 1:N
            v_rms_arr[j, i] = sqrt.(sum(v_sq_arr[j, 2:i] .* diff(t_arr[1:i]))/t_arr[i])
        end
    end

    return v_rms_arr
end

function kinetic_energy()
    return 0.5 * sum(mass .* sum(velocity.^2, dims=2))

end;

function get_blast_radius()
    # temp_center = [0.0, 0.0]
    temp_radius = 0.0
    count = 0

    for i in 1:N
        if KE_arr[i] != 0.0
            temp_radius += norm(position[i, :] - center[:])
            count += 1
        end

    end

    return temp_radius/count

end; # function get_blast_radius ends here

function get_blast_center()
    temp_center = [0.0, 0.0]
    count = 0

    for i in 1:N
        if KE_arr[i] != 0.0
            temp_center += position[i, :]
            count += 1
        end

    end

    return temp_center/count

end; # function get_blast_center ends here


"""
Returns the density of particles within a circle of radius r.
"""
function get_density_within(r)

    density = 0.0
    c = simbox/2

    for i in 1:N

        if norm(p_arr[i].pos - c) < r
            density += p_arr[i].m
        end

    end

    return density / (pi*r^2)

end;


###########################################################################
########################### Plotting Routines #############################
###########################################################################

"""
Returns the absolute position in the simulation box for given
relative position.
modified on 26/01/2021.
run.jl is not compatible with this function.
"""
function get_abs_pos(xr,yr,frame_xlim,frame_ylim)
    x = xr * (frame_xlim[2]-frame_xlim[1]) + frame_xlim[1]
    y = yr * (frame_ylim[2]-frame_ylim[1]) + frame_ylim[1]
    return x, y

end; # function get_abs_pos ends here


"""
Plots a 2D particle as circle.
"""
function circle(x, y, r; n=30)

    θ = 0:360÷n:360
    Plots.Shape(r*sind.(θ) .+ x, r*cosd.(θ) .+ y)

end; # function circle ends here


"""
Annotates the particle with its name-tag.
"""
function nameParticle(x,y,i)
    annotate!(x, y, text(L"%$(i)", 12), color=:red)

end; # function nameParticle ends here


"""
Plot cells.
"""
function plot_cells()

    # hline!([0:cell_size:frame_xlim[2]]; color=:black, linewidth=0.5, alpha=0.5)
    # vline!([0:cell_size:frame_ylim[2]]; color=:black, linewidth=0.5, alpha=0.5)

    hline!([0:cell_size:simbox[1]]; color=:black, linewidth=0.5, alpha=0.5)
    vline!([0:cell_size:simbox[2]]; color=:black, linewidth=0.5, alpha=0.5)

    hline!([simbox[1]]; color=:black, linewidth=0.5)
    vline!([simbox[2]]; color=:black, linewidth=0.5)

end

"""
Plot box.
"""
function plot_box()

    plot!([0, simbox[1]], [0, 0]; color=:black, linewidth=1, ls=:solid)
    plot!([0, simbox[1]], [simbox[2], simbox[2]]; color=:black, linewidth=1, ls=:solid)
    plot!([0, 0], [0, simbox[2]]; color=:black, linewidth=1, ls=:solid)
    plot!([simbox[1], simbox[1]], [0, simbox[2]]; color=:black, linewidth=1, ls=:solid)

end

"""
Puts the cell info in the plot.
"""
function cell_info()
    n_cells = Int64.(ceil.(simbox/cell_size))
    txt = ""
    for i in 1:n_cells[1]
        for j in 1:n_cells[2]
            if !isnothing(cell_list[i,j])
                p = cell_list[i,j]
                txt *= "Cell-($(i), $(j)): "
                while !isnothing(p.next)
                    txt *= "$(p.id), "
                    p = p.next
                end
                txt *= "$(p.id)\n"

            end
        end
    end

    tagLoc = get_abs_pos(0.02, 0.8, frame_xlim, frame_ylim)
    annotate!(tagLoc[1], tagLoc[2], text(txt, 7, valign=:top, halign=:left), color=:black, fontfamily="Helvetica")
end


"""
Plots single frame of the simulation.
"""
function plot_frame(fname)

    plot() # clear the plot

    # nKE = KE_arr ./ E_expl

    clrs = []
    for pke in KE_arr
        if pke == 0.0
            push!(clrs, :blue)
        else
            push!(clrs, :red)
        end
    end

    circles = circle.(position[:, 1], position[:, 2], radii)

    plot_kwargs = (
        color=permutedims(clrs),
        # color=:blue,
        aspect_ratio=:equal, fontfamily="Helvetica", legend=false, line=nothing,
        xlim=frame_xlim, ylim=frame_ylim,
        # xlim=(0,simbox[1]), ylim=(0,simbox[2]),
        framestyle=:box,
        # size=(1024,1024),
        size=(512,512),
        # xticks=collect(0:2:simbox[1]+1), yticks=collect(0:2:simbox[2]+1),
        grid=false,# gridalpha=0.5, gridcolor=:black,
        xformatter = :none, yformatter = :none
        )

    if draw_cells
        plot_cells() # comment this line to remove cell lines
    end

    if draw_box
        plot_box()
    end

    # println("Plotting particles.")

    plot!(circles; plot_kwargs...)
    # scatter!(position[:, 1], position[:, 2]; ms=1, plot_kwargs...)

    if plot_blast_front
        plot!(circle(simbox[1]/2, simbox[2]/2, R(sim_time[1]); n=100); linecolor=:black, linewidth=2, fillalpha=0)
    end

    if annotate_particles
        nameParticle.(position[:, 1], position[:, 2], 1:N)
    end

    tagLoc = get_abs_pos(0.02, 0.99, frame_xlim, frame_ylim)
    txt = "t=$(round(sim_time[end], digits=2))\n"
    txt *= "event count = $(event_num[1])\n"
    txt *= "collision count = $(col_num[1])"

    annotate!(tagLoc[1], tagLoc[2], text(txt, 10, valign=:top, halign=:left),
    color=:black, fontfamily="Helvetica")

    # if col_num[1] != 0
    #     print("; Plotting blast front.")
    #     blast_radius = get_blast_radius()
    #     blast_center = get_blast_center()
    #     blast_front = circle(center[1], center[2], blast_radius; n=100)
    #     plot!(blast_front; linecolor=:black, linewidth=5, fillalpha=0)

    #     blast_front = circle(blast_center[1], blast_center[2], blast_radius; n=100)
    #     plot!(blast_front; linecolor=:purple, linewidth=5, fillalpha=0)

    #     push!(btime, sim_time[1])
    #     push!(bradius, blast_radius)

    # end

    # cell_info() # comment this line to remove cell info

    savefig(fname)
    # savefig("./Images/N=$N-frame-$(div(event_num[1], 100)).png")
    # savefig("./N=$N-col-num=$(col_num[1]).png")
    # savefig("./Images/N=$N-frame-$(event_num[1]).png")
    # savefig(imgloc * fname)

end;

function R(t)
    return (E_expl/(A * rho_inf))^0.25 * t^0.5
end;


# Plotting section ends here
# ===========================================================================
#


# function save_data(df_name)

#     typs = []

#     for pke in KE_arr
#         if pke == 0.0
#             push!(typs, 1)
#         else
#             push!(typs, 2)
#         end
#     end

#     df = open(df_name, "w")
#     txt = "$N\n"
#     txt *= "type radius x y vx vy\n"
#     for i in 1:N

#         txt *= "$(typs[i]) $(radii[i]) $(position[i, 1]) $(position[i, 2]) $(velocity[i, 1]) $(velocity[i, 2])\n"

#     end

#     write(df, txt)

# end


function save_data(df_name)

    df = open(df_name, "w")
    txt = "$N\n"
    txt *= "radius x y vx vy\n"

    for i in 1:N

        txt *= "$(radii[i]) $(position[i, 1]) $(position[i, 2]) $(velocity[i, 1]) $(velocity[i, 2])\n"

    end

    write(df, txt)

end
