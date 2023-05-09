#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def IdentifyVertsAndCells(i,cell_verts,vert_cells,vert_neighs,T1_edge,l):
    #Find vertices undergoing T1
    if T1_edge == len(l) - 1:
        T1_verts = cell_verts[i,[T1_edge,0]]
    else:
        T1_verts = cell_verts[i,[T1_edge,T1_edge+1]]
    #Identify the four cells are affected by transition
    dummy = np.concatenate((vert_cells[T1_verts[0]],vert_cells[T1_verts[1]]))
    T1cells = np.unique(dummy)

    #Identify cells that are neighbours prior to transition (that won't be afterwards)
    old_neigh = np.intersect1d(vert_cells[T1_verts[0]],vert_cells[T1_verts[1]])
    
    #Identify cells that will be neighbours after transition
    notneigh1 = T1cells[T1cells != old_neigh[0]]
    notneigh2 = T1cells[T1cells != old_neigh[1]]
    new_neigh = np.intersect1d(notneigh1,notneigh2)
    
    old_vert_neighs = vert_neighs[T1_verts,:]
    
    return T1_verts, old_neigh, new_neigh, old_vert_neighs

def UpdateVertCoords(i,verts,points,cell_vert_coords,old_neigh,T1_verts,T1_thresh,l,T1_edge,x_max,y_max):
    #Define point from which to rotate vertex coords around
    rotation_centre = cell_vert_coords[T1_edge,:] + 0.5*l[T1_edge,:]
    
    #Find direction of vector between old_neigh centroids (new vertex positions will 
    #be on a line parallel to this vector, passing through the rotation centre)
    old_neigh_dist = points[old_neigh[1],:] - points[old_neigh[0],:]
    shifted = np.nonzero(abs(old_neigh_dist) > 4)
    for j in range(len(shifted[0])):
        if shifted[0][j] == 0:
            if old_neigh_dist[0] < 0:
                old_neigh_dist[0] += x_max
            else:
                old_neigh_dist[0] -= x_max
        else:
            if old_neigh_dist[1] < 0:
                old_neigh_dist[1] += y_max
            else:
                old_neigh_dist[1] -= y_max

    old_neigh_direction = old_neigh_dist / np.linalg.norm(old_neigh_dist)
    
    #Update vertex positions, with a new edge length = 1.5*T1_thresh and such that 
    #T1_verts[0] is always furthest from old_neigh[0]. Doing this makes updating 
    #vert_cells, vert_neighs and cell_verts easier
    verts[T1_verts[0],:] = rotation_centre + 0.75*T1_thresh*old_neigh_direction
    verts[T1_verts[1],:] = rotation_centre - 0.75*T1_thresh*old_neigh_direction
    
    return verts

def UpdateVertCells(verts,points,vert_cells,T1_verts,old_neigh,new_neigh):
    #Update vert_cells of T1 verts
    vert_cells[T1_verts[0],:] = np.array([new_neigh[0],new_neigh[1],old_neigh[1]])
    vert_cells[T1_verts[1],:] = np.array([new_neigh[0],new_neigh[1],old_neigh[0]])
        
    return vert_cells
    
def UpdateCellVerts(verts,points,cell_verts,T1_verts,old_neigh,new_neigh,N):
    #Remove lost vertex from cell_verts of old_neigh
    for j in range(2):
        to_delete =  np.nonzero(cell_verts[old_neigh[j],:]==T1_verts[j])
        cell_verts[old_neigh[j],to_delete[0][0]:-1] = cell_verts[old_neigh[j],to_delete[0][0]+1:]
        cell_verts[old_neigh[j],-1] = 2*N

    #Insert new vertex to cell_verts of new_neigh (such that cell_verts remain in a clockwise order)
    missing_vert = np.nonzero(cell_verts[new_neigh[0],:] == T1_verts[0])
    for j in range(2):
        if len(missing_vert[0]) == 0:               
            vert_pos =  np.nonzero(cell_verts[new_neigh[j],:]==T1_verts[j-1])
            cell_verts[new_neigh[j],vert_pos[0][0]+2:] = cell_verts[new_neigh[j],vert_pos[0][0]+1:-1]
            cell_verts[new_neigh[j],vert_pos[0][0]+1] = T1_verts[j]
        else:
            vert_pos =  np.nonzero(cell_verts[new_neigh[j-1],:]==T1_verts[j-1])
            cell_verts[new_neigh[j-1],vert_pos[0][0]+2:] = cell_verts[new_neigh[j-1],vert_pos[0][0]+1:-1]
            cell_verts[new_neigh[j-1],vert_pos[0][0]+1] = T1_verts[j]
        
    return cell_verts
        
def UpdateVertNeighs(vert_neighs,points,cell_verts,T1_verts,old_neigh,new_neigh,old_vert_neighs,N):
    #Update vert_neighs of T1 verts such that each one contains the other T1 vert and the vertices 
    #immediately clockwise and anticlockwise to it in cell_verts of old_neigh
    for j in range(2):
        no_of_verts = np.nonzero(cell_verts[old_neigh[j],:] < 2*N)
        vert_pos = np.nonzero(cell_verts[old_neigh[j],no_of_verts[0]] == T1_verts[j-1])
        cell_verts_old_neigh_up = np.roll(cell_verts[old_neigh[j],no_of_verts[0]],1)
        cell_verts_old_neigh_down = np.roll(cell_verts[old_neigh[j],no_of_verts[0]],-1)
        vert_neighs[T1_verts[j-1],:] = np.array([T1_verts[j],cell_verts_old_neigh_up[vert_pos[0][0]],cell_verts_old_neigh_down[vert_pos[0][0]]])
    
    #Update other vert_neighs affected by T1. This entails exchanging one T1_vert for another 
    #in the vert_neighs of two of the vertices connected to the T1_verts
    for j in range(2):
        verts2notalter = np.intersect1d(old_vert_neighs[j,:],vert_neighs[T1_verts[j],:])
        notvert1 = np.nonzero(vert_neighs[T1_verts[j],:] != verts2notalter[0])
        notvert2 = np.nonzero(vert_neighs[T1_verts[j],:] != verts2notalter[1])
        
        verts2alterind = np.intersect1d(notvert1,notvert2)
        verts2alter = vert_neighs[T1_verts[j],verts2alterind[0]]
        
        neigh2alter = np.argwhere(vert_neighs[verts2alter,:] == T1_verts[j-1])
        vert_neighs[verts2alter,neigh2alter[0]] = T1_verts[j]
    
    return vert_neighs