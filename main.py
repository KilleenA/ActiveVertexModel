# -*- coding: utf-8 -*-
import numpy as np
import functions as f

#Define Parameters
N = 64                      #Number of cells
it = 0                      #Iteration counter
N_t = 3e3                   #Number of time-steps
dt = 0.01                   #Time-step
N_init = 1000               #Number of initial time-steps 
N_relax = 1000              #Number of relaxation time-steps 
save_it = 10                #Save interval
T1_thresh = 0.01            #Edge length to initiate T1 transition
d = np.sqrt(2/np.sqrt(3))   #Initial x spacing between cell centres
h = np.sqrt(3) * d/2        #Initial y spacing between cell centres
l = d/np.sqrt(3)            #Initial cell edge length
x_max = d * np.sqrt(N)      #Maximum x coordinate
y_max = h * np.sqrt(N)      #Maximum y coordinate
f0_init = 0.5               #Self propulsion magnitude for initialisation
P0_init = 3.7               #Preferred perimeter for initialisation

f0_sim = 1.0                #Self propulsion magnitude for simulation
P0_sim = 3.8                #Preferred perimeter for simulation
A0 = 1                      #Preferred area
Ka = 1                      #Area stiffness
Kp = 1                      #Perimeter stiffness
mu = 1                      #Friction coefficient
Dr = 1                      #Rotational diffusion constant

#Seed initial points to generate initial voronoi tessellation
points = f.Seeding(N,d,h)
#Generate initial voronoi tessellation to initialise vertex list
verts = f.PeriodicVoronoi(points,N,x_max,y_max)
#Generate list of which vertices belong to each cell
cell_verts = f.FindCellVerts(points,verts,d,x_max,y_max,N)
#Generate list of which three cells neighbour each vertex
vert_cells = f.FindVertCells(points,verts,d,x_max,y_max,N)
#Generate list of which three vertices are connected to each vertex
vert_neighs = f.FindVertNeighs(verts,l,d,x_max,y_max,N)
#Initialise cell polarities
polarities = np.random.uniform(-np.pi,np.pi,(N,1))

while it < N_t: 
    #Initialise system by running N_init time-steps in a fluid state to
    #create random tissue configuration. Then turn off activity to relax system
    #to equilibrium. Then start simulation proper.     
    if it < N_init:
        f0 = f0_init
        P0 = P0_init
    elif it < (N_init+N_relax):
        f0 = 0
        P0 = P0_sim
    else:
        f0 = f0_sim
        P0 = P0_sim
        
    #Calculate position of vertices in each cell
    cell_vert_coords = f.FindCellVertCoords(points,verts,cell_verts,N,x_max,y_max)
    #Calculate area and perimeter of each cell
    A, P = f.CalculateAandP(cell_vert_coords,N)
    #Calculate 'passive' force on each vertex
    f_pass = f.CalculatePassiveForces(vert_neighs, vert_cells, cell_verts, verts, A, P, A0, P0, Ka, Kp, x_max, y_max, N);
    #Calculate active force on each vertex
    f_act = f.CalculateActiveForces(vert_cells,polarities,N,f0)
    #Update vertex, centroid positions, polarity vectors and vertex positions in cell_vert_coords
    verts, polarity, points = f.UpdatePosPol(verts,cell_verts,points,polarities,N,mu,Dr,dt,f_pass,f_act,x_max,y_max)
    #Update topology via T1 transitions
    verts, vert_neighs, vert_cells, cell_verts = f.UpdateTopology(points,verts,vert_neighs,vert_cells,cell_verts,cell_vert_coords,l,x_max,y_max,T1_thresh,N)
    #Implement periodic boundary conditions
    verts,points = f.PeriodicBoundaries(verts,points,N,x_max,y_max)

    #Save data
    if it % save_it == 0:
        it_str = "%d" %(it)
        
        filename = "./PointFiles/points" + it_str.zfill(6) + ".txt"
        np.savetxt(filename, points)
        
        filename = "./VertFiles/verts" + it_str.zfill(6) + ".txt"
        np.savetxt(filename, verts)
        
        f.SaveCells(cell_vert_coords, points, verts, N, x_max, y_max, it)
        
        print("finished %d iterations" %(it))

    it += 1

    
    