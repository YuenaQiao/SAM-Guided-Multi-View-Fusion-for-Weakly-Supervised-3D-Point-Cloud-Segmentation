import cupy as cp
import cupyx.scipy.sparse as cp_sparse
from cupyx.scipy.sparse import diags

class HyperGraph():

    def __init__(self, H):
        self.sparse_H = H

    def LabelPropagation_Subset(self, subset_idx, Y):
        if len(subset_idx) != 0:
            sparse_H_subset = self.sparse_H[subset_idx]
        else:
            sparse_H_subset = self.sparse_H
            
        edge_degrees = cp_sparse.csr_matrix(sparse_H_subset.sum(axis=0))

        De = diags(edge_degrees.A.ravel())

        inverse_diagonal_elements = cp.where(De.diagonal() != 0, 1.0 / De.diagonal(), 0)
        De_inv = diags(inverse_diagonal_elements)

        vertex_degrees = cp_sparse.csr_matrix(sparse_H_subset.sum(axis=1))


        Dv_inv_sqrt = diags((1 / vertex_degrees.A.ravel()) ** (0.5))
        Dv_inv_sqrt_sparse = cp_sparse.csc_matrix(Dv_inv_sqrt)

        DvH = Dv_inv_sqrt_sparse.dot(sparse_H_subset)
        HDv = DvH.transpose()
        DvHDe = DvH.dot(De_inv)
        DvHDeHDv = DvHDe.dot(HDv)

        hyper_lambda = 1
        F_pre = (1+1/hyper_lambda) * cp.eye(DvHDeHDv.shape[0]) - (1/hyper_lambda)*DvHDeHDv
        F_pre_inv = cp.linalg.inv(F_pre)

        Y_subset_cupy = cp.asarray(Y[subset_idx])

        Y_hat = F_pre_inv.dot(Y_subset_cupy)

        return Y_hat

