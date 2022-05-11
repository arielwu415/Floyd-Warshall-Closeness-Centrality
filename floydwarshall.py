import sys
import numpy as np
from mpi4py import MPI


# @param adj_local:       The sub-matrix (numpy ndarray) that each processor has
# @param n:               The original matrix size, or total number of nodes
# @param rows_per_proc:   A list contains the numbers of rows each processor has
# @param comm:            The mpi4py communication protocol
def floyd_warshall(adj_local, n, rows_per_proc, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    for t in range(0, size):
        sys.stdout.write("Task:{}; Processor {}\n".format(t, rank))
        adj_rproc = adj_local
        rproc = (rank - t + size) % size

        if t != 0:
            sproc = (rank + t) % size

            adj_rproc = np.empty(shape=(n, rows_per_proc[rproc]), dtype='i')

            sreq = comm.Isend([adj_local, MPI.INT], dest=sproc, tag=sproc)
            rreq = comm.Irecv([adj_rproc, MPI.INT], source=rproc, tag=rank)
            sreq.wait()
            rreq.wait()

        local_num_rows = rows_per_proc[rank]
        rproc_num_rows = rows_per_proc[rproc]

        row_offset = sum(rows_per_proc[:rproc])

        for k in range(0, rproc_num_rows):
            for i in range(0, local_num_rows):
                for j in range(0, n):
                    adj_local[i, j] = min(adj_local[i, j], adj_local[i, k + row_offset], adj_rproc[k, j])
