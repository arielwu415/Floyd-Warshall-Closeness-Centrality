# -*- coding: utf-8 -*-
"""
Created on Tue May 10 20:57:24 2022

@author: Ariel
"""

from mpi4py import MPI
import numpy as np


# @param adj_local:       The sub-matrix (numpy ndarray) that each processor has
# @param n:               The original matrix size, or total number of nodes
# @param rows_per_proc:   A list contains the numbers of rows each processor has
# @param comm:            The mpi4py communication protocol
def calculate_closeness_centrality(adj_local, n, rows_per_proc, comm):
    
    # Get rank of each processor
    rank = comm.Get_rank()
    
    # eg.
    # row_num = [3, 2]
    # | a00 a01 a02 a03 a04 |
    # | a10 a11 a12 a13 a04 | Q0
    # | a20 a21 a22 a23 a04 |
    # -----------------------
    # | a30 a31 a32 a33 a04 | 
    # | a40 a41 a42 a43 a04 | Q1
    
    ### Create a numpy array to store the centrality values
    Cr = np.zeros(rows_per_proc[rank])
    
    ### Iterate through all rows in one processor
    for i in range(rows_per_proc[rank]):
        
        # sum up each row except the self node to get the total distance
        total_dist = sum(adj_local[i,:]) - adj_local[i,i]
        centrality = 1 / (total_dist - 1)
        
        Cr[i] = centrality
    