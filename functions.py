#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import Voronoi
import T1functions as T1f

def Seeding(N,d,h):
    #Seed points to create a hexagonal lattice, this means the initial voronoi 
    #tesselation will be a grid of regular hexagons with side length l
    #This is to make initialising cell_verts, verts_cells and vert_neighs easier
    points = np.zeros((N,2))
    L = int(np.sqrt(N))
    for i in range(L):
        if i % 2 == 0:
            offset = 0.25 * d
        else:
            offset = 0.75 * d
        
        for j in range(L):
            points[i * L+j,0] = offset + (j * d)
            points[i * L+j,1] = 0.5 * h + (i * h)
            
    return points
            
def PeriodicVoronoi(points,N,x_max,y_max):
    #mirror points to create a domain 3x3 larger than actually required
    #(with actual domain in the centre of 3x3 grid)
    points_mirrored  = np.zeros((9*N,2))
    shift = np.array([[-x_max,y_max],[0,y_max],[x_max,y_max],[-x_max,0],[0,0],[x_max,0],[-x_max,-y_max],[0,-y_max],[x_max,-y_max]])
    for i in range(9):
        points_mirrored[i*N:(i+1)*N,:] = points + shift[i,:]
    
    #Generate voronoi diagram of this domain (using 3x3 grid ensures correct periodicity)
    mirrored_vor = Voronoi(points_mirrored)
    
    #Extract vertices that lie within actual simulation domain
    vertices_x = np.where(np.logical_and(mirrored_vor.vertices[:,0] >= 0, mirrored_vor.vertices[:,0] <= x_max))
    vertices_y = np.where(np.logical_and(mirrored_vor.vertices[:,1] >= 0, mirrored_vor.vertices[:,1] <= y_max))
    vert_indices = np.intersect1d(vertices_x,vertices_y)
    
    verts = mirrored_vor.vertices[vert_indices,:]
    return verts

def FindCellVerts(points,verts,d,x_max,y_max,N):
    #Initialise cell_verts to allow up to 10 vertices per cell (Although initially each cell has 6)
    #Initialise all elements to be 2*N to follow convention of other data structures
    cell_verts = 2*N*np.ones((N,10),dtype=int) 
    for i in range(N):
        #Find vertices at a distance < d from cell centroid (as these must belong to cell i)  
        rel_dist = verts - points[i,:]
        sq_rel = np.linalg.norm(rel_dist,axis=1)
        vert_inds = np.nonzero(sq_rel < d)[0]
        coords = verts[sq_rel < d,:]
        
        #Find vertices that were intially missed due to periodic BC
        shift = np.array([[-x_max,y_max],[0,y_max],[x_max,y_max],[-x_max,0],[x_max,0],[-x_max,-y_max],[0,-y_max],[x_max,-y_max]])
        for j in range(len(shift)):
            shift_rel_dist = rel_dist + shift[j,:]
            sq_rel = np.linalg.norm(shift_rel_dist,axis=1)
            extra_verts = np.nonzero(sq_rel < d)
            if np.shape(extra_verts)[1] > 0:
                vert_inds = np.concatenate((vert_inds,extra_verts),axis=None)
        
        #Find the coordinates of cell vertices so that cell vertex indices can be put in a clockwise order
        coords = verts[vert_inds,:]
        for j in range(6):
            if  (coords[j,0] - points[i,0])  > 4:
                coords[j,0] -= x_max
            elif (coords[j,0] - points[i,0]) < -4:
                coords[j,0] += x_max

            if (coords[j,1] - points[i,1]) > 4:
                coords[j,1] -= y_max
            elif (coords[j,1] - points[i,1]) < -4:
                coords[j,1] += y_max
        
        #Put vertices in a clockwise order 
        inds_dummy = np.zeros(6)
        arg = np.arctan2((coords[:,1]-points[i,1]),(coords[:,0]-points[i,0]))
        indices = np.argsort(arg)
        for j in range(6):
            inds_dummy[j] = vert_inds[indices[j]]

        inds_dummy = np.flipud(inds_dummy) #atan2 measures angles anti-clockwise so must flip to get clockwise order
        cell_verts[i,:6] = np.transpose(inds_dummy)
        
    return cell_verts

def FindVertCells(points,verts,d,x_max,y_max,N):
    #Initialise all elements to be 2*N because there are 2*N vertices in model, which are therefore indexed from 0 to 2*N-1
    vert_cells = np.zeros((2*N,3),dtype=int) 
    
    for i in range(2*N):
        #Find cells at a distance < d from vertex i (as these must neighbour vertex i)  
        rel_dist = points-verts[i,:]
        sq_rel = np.linalg.norm(rel_dist,axis=1)
        vert_inds = np.nonzero(sq_rel < d)[0]
        
        #Find cells that were intially missed due to periodic BC
        shift = np.array([[-x_max,y_max],[0,y_max],[x_max,y_max],[-x_max,0],[x_max,0],[-x_max,-y_max],[0,-y_max],[x_max,-y_max]])
        for j in range(len(shift)):
            shift_rel_dist = rel_dist + shift[j,:]
            sq_rel = np.linalg.norm(shift_rel_dist,axis=1)
            extra_verts = np.nonzero(sq_rel < d)
            if np.shape(extra_verts)[1] > 0:
                vert_inds = np.concatenate((vert_inds,extra_verts),axis=None)
        
        vert_cells[i,:] = np.transpose(vert_inds)
        
    return vert_cells

def FindVertNeighs(verts,l,d,x_max,y_max,N):
    #Initialise all elements to be 2*N because there are 2*N vertices in model, which are therefore indexed from 0 to 2*N-1
    vert_neighs = np.zeros((2*N,3),dtype=int) 
    
    for i in range(2*N):
        #Find vertices at a distance == l (+- 1e-12) from vertex i (as these must neighbour vertex i) 
        rel_dist = verts-verts[i,:]
        sq_rel = np.linalg.norm(rel_dist,axis=1)
        vert_inds = np.argwhere(abs(sq_rel-l) < 1e-12)
        
        #Find vertices that were intially missed due to periodic BC
        shift = np.array([[-x_max,y_max],[0,y_max],[x_max,y_max],[-x_max,0],[x_max,0],[-x_max,-y_max],[0,-y_max],[x_max,-y_max]])
        for j in range(len(shift)):
            shift_rel_dist = rel_dist + shift[j,:]
            sq_rel = np.linalg.norm(shift_rel_dist,axis=1)
            extra_verts = np.nonzero(abs(sq_rel-l) < 1e-12)
            if np.shape(extra_verts)[1] > 0:
                vert_inds = np.concatenate((vert_inds,extra_verts),axis=None)
                
        vert_neighs[i,:] = np.transpose(vert_inds)

    return vert_neighs

def FindCellVertCoords(points,verts,cell_verts,N,x_max,y_max):
    #Initialise to allow up to ten vertices per cell
    cell_vert_coords = np.zeros((N,10,2))
    for i in range(N):
        #Find vertices and cell i and vertex coords (not accounting for periodicity)
        vert_inds = cell_verts[i,:] < 2*N
        cell_i = verts[cell_verts[i,vert_inds],:]
        
        #Shift vertex coords to account for periodic BC
        rel_dists = cell_i - points[i,:]
        shifted = np.nonzero(abs(rel_dists) > 4)
        for j in range(len(shifted[0])):
            if shifted[1][j] == 0:
                if rel_dists[shifted[0][j],shifted[1][j]] < 0:
                    cell_i[shifted[0][j],shifted[1][j]] += x_max
                else:
                    cell_i[shifted[0][j],shifted[1][j]] -= x_max
            else:
                if rel_dists[shifted[0][j],shifted[1][j]] < 0:
                    cell_i[shifted[0][j],shifted[1][j]] += y_max
                else:
                    cell_i[shifted[0][j],shifted[1][j]] -= y_max
        cell_vert_coords[i,:len(cell_i),:] = cell_i
        
    return cell_vert_coords

def CalculateAandP(cell_vert_coords,N):
    A = np.zeros((N,1))
    P = np.zeros((N,1))
    for i in range(N):
        #Determine how many vertices are in cell i and extract their coordinates
        no_of_verts = np.nonzero(cell_vert_coords[i,:,:])
        verts = cell_vert_coords[i,:no_of_verts[0][-1]+1,:]
        shifted_verts = np.roll(verts,1,axis=0)
        
        #Find area of cell i using the shoelace method
        shoelace = np.dot(verts[:,0], shifted_verts[:,1]) - np.dot(verts[:,1], shifted_verts[:,0]);
        A[i] = 0.5 * abs(shoelace)
        
        #Find perimeter of cell i by summing edge lengths
        l =  shifted_verts - verts
        P[i] = sum(np.linalg.norm(l,axis=1))

    return A, P
    
def CalculatePassiveForces(vert_neighs, vert_cells, cell_verts, verts_orig, A, P, A0, P0, Ka, Kp, x_max, y_max, N):
    f_pass = np.zeros((2*N,2))
    for i in range(2*N):
        verts = verts_orig #Create a dummy copy of vertex coords
        dEdr = np.zeros((3,2))
        
        #Shift vertex coords that may have incorrect position in relation to vertex i due to periodicity
        int_dists = verts[vert_neighs[i,:],:] - verts[i,:]
        shifted = np.nonzero(abs(int_dists) > 4); 
        for j in range(len(shifted[0])):
            if shifted[1][j] == 0:
                if int_dists[shifted[0][j],shifted[1][j]] < 0:
                    verts[vert_neighs[i,shifted[0][j]],shifted[1][j]] += x_max
                else:
                    verts[vert_neighs[i,shifted[0][j]],shifted[1][j]] -= x_max
            else:
                if int_dists[shifted[0][j],shifted[1][j]] < 0:
                    verts[vert_neighs[i,shifted[0][j]],shifted[1][j]] += y_max
                else:
                    verts[vert_neighs[i,shifted[0][j]],shifted[1][j]] -= y_max
        
        #Calculate contribution to passive force on vertex i from each of three cells, j, that neighbour it
        for j in range(3):
            curr_cell = vert_cells[i,j]

            #Find vertices belonging to cell j
            no_of_verts_vec = np.nonzero(cell_verts[curr_cell,:] < 2*N)
            curr_cell_verts = cell_verts[curr_cell,no_of_verts_vec]
            
            #dPdr requires the vectors from vertex i to adjacent vertices in cell j
            #(immediately anticlokwise and clockwise from vertex i)
            vert_i_pos = np.nonzero(curr_cell_verts == i)
            vert_anticlock = np.roll(curr_cell_verts,1)
            vert_clock = np.roll(curr_cell_verts,-1)
            r = np.zeros((2,2))
            r[0,:] = verts[i,:] - verts[vert_anticlock[vert_i_pos[0],vert_i_pos[1]],:]
            r[1,:] = verts[i,:] - verts[vert_clock[vert_i_pos[0],vert_i_pos[1]],:]
            
            #dAdr requires the outward normal vectors of these two vectors
            n = np.zeros((2,2))
            n[0,:] = verts[i,:] - verts[vert_anticlock[vert_i_pos[0],vert_i_pos[1]],:]
            n[1,:] = verts[i,:] - verts[vert_clock[vert_i_pos[0],vert_i_pos[1]],:] 
            n = np.fliplr(n)
            n[0,0] = -n[0,0]
            n[1,1] = -n[1,1]
            
            dAdr = n[0,:] + n[1,:]
            dPdr = (r[0,:]/np.linalg.norm(r[0,:])) + (r[1,:]/np.linalg.norm(r[1,:])) 
            
            dEdr[j,:] = Ka*(A[curr_cell]-A0)*dAdr + 2*Kp*(P[curr_cell]-P0)*dPdr

        f_pass[i,:] = np.sum(-dEdr,axis=0)
                
    return f_pass

def CalculateActiveForces(vert_cells,polarities,N,f0):
    #Calculate the active force vector for each cell
    polarities_vec = np.concatenate((np.cos(polarities),np.sin(polarities)),axis=1)
    f_act_cell = f0 * polarities_vec
    
    f_act = np.zeros((2*N,2))
    for i in range(2*N):
        #Active force on vertex i is the average active force from 3 cells that neighbour it
        f_act[i,:] = np.sum(f_act_cell[vert_cells[i,:],:],axis=0) / 3.0;

    return f_act

def UpdatePosPol(verts,cell_verts,points,polarities,N,mu,Dr,dt,f_pass,f_act,x_max,y_max):
    verts += dt * mu * (f_pass + f_act)
    polarities += np.sqrt(2*Dr*dt) * np.random.normal(0,1,(N,1));
    
    #find vertex positions using cell_verts_coords and use to update positions of cell centroids
    cell_vert_coords = FindCellVertCoords(points,verts,cell_verts,N,x_max,y_max)
    for i in range(N):
        vert_inds = cell_verts[i,:] < 2*N
        points[i,:] = np.mean(cell_vert_coords[i,vert_inds,:],axis=0)
    
    return verts, polarities, points
    
def UpdateTopology(points,verts,vert_neighs,vert_cells,cell_verts,cell_vert_coords,l,x_max,y_max,T1_thresh,N):
    for i in range(N):
        #Determine how many vertices are in cell i
        vert_inds = cell_verts[i,:] < 2*N
        
        #If cell i only has three sides don't perform a T1 on it (you can't have a cell with 2 sides) 
        if(len(vert_inds) == 3):
            continue 
        
        #Find UP TO DATE vertex coords of cell i
        vert_inds = cell_verts[i,:] < 2*N
        cell_i = verts[cell_verts[i,vert_inds],:]

        rel_dists = cell_i - points[i,:]
        shifted = np.nonzero(abs(rel_dists) > 4)
        for j in range(len(shifted[0])):
            if shifted[1][j] == 0:
                if rel_dists[shifted[0][j],shifted[1][j]] < 0:
                    cell_i[shifted[0][j],shifted[1][j]] += x_max
                else:
                    cell_i[shifted[0][j],shifted[1][j]] -= x_max
            else:
                if rel_dists[shifted[0][j],shifted[1][j]] < 0:
                    cell_i[shifted[0][j],shifted[1][j]] += y_max
                else:
                    cell_i[shifted[0][j],shifted[1][j]] -= y_max
        
        #Calculate the lengths, l, of each cell edge
        shifted_verts = np.roll(cell_i,-1,axis=0)
        l =  shifted_verts - cell_i
        l_mag = np.linalg.norm(l,axis=1)
        #Determine if any edges are below threshold length
        to_change = np.nonzero(l_mag < T1_thresh)
        
        if len(to_change[0]) == 0:
            continue
        else:
            T1_edge = to_change[0][0]
            #Identify vertices and cells affected, also return vert_neighs of old neighbours (to be used when updated vert_neighs later on)
            T1_verts, old_neigh, new_neigh, old_vert_neighs = T1f.IdentifyVertsAndCells(i,cell_verts,vert_cells,vert_neighs,T1_edge,l)
            #Update vertex coordinates
            verts = T1f.UpdateVertCoords(i,verts,points,cell_i,old_neigh,T1_verts,T1_thresh,l,T1_edge,x_max,y_max)    
            #Update vert_cells
            vert_cells = T1f.UpdateVertCells(verts,points,vert_cells,T1_verts,old_neigh,new_neigh)
            #Update cell_verts 
            cell_verts = T1f.UpdateCellVerts(verts,points,cell_verts,T1_verts,old_neigh,new_neigh,N)
            #Update vert_neighs
            vert_neighs = T1f.UpdateVertNeighs(vert_neighs,points,cell_verts,T1_verts,old_neigh,new_neigh,old_vert_neighs,N)
            
    return verts, vert_neighs, vert_cells, cell_verts
    
def PeriodicBoundaries(verts,points,N,x_max,y_max):
    for i in range(2*N):
        #Apply periodic BC to centroids
        if i < N:
            if points[i,0] > x_max:
                points[i,0] -= x_max
            elif points[i,0] < 0:
                points[i,0] += x_max
    
            if points[i,1] > y_max:
                points[i,1] -= y_max
            elif points[i,1] < 0:
                points[i,1] += y_max
        #Apply periodic BC to vertices
        if verts[i,0] > x_max:
            verts[i,0] -= x_max
        elif verts[i,0] < 0:
            verts[i,0] += x_max

        if verts[i,1] > y_max:
            verts[i,1] -= y_max
        elif verts[i,1] < 0:
            verts[i,1] += y_max
            
    return verts,points
    
def SaveCells(cell_vert_coords, points, verts, N, x_max, y_max, it):
    it_str = "%d" %it
    
    filename = "./CellFiles/cells" + it_str.zfill(6) + ".gnu"
    f = open(filename, "w")

    for i in range(N):
        no_of_cell_verts = np.argwhere(cell_vert_coords[i,:,0] != 0)
        
        #To create closed polygons must add first vertex to end of vertices for cell i
        coords_to_save = np.concatenate((cell_vert_coords[i,:len(no_of_cell_verts),:],np.expand_dims(cell_vert_coords[i,0,:],axis=0)), axis=0)
        for j in range(len(coords_to_save)):
            if (j+1) % len(coords_to_save) == 0:
                #If vertex is last to be written in cell leave blank line in the file
                f.write("%g %g\n\n" %(coords_to_save[0,0], coords_to_save[0,1]))
            else:
                f.write("%g %g\n" %(coords_to_save[j,0], coords_to_save[j,1]))

    f.close()

    
    