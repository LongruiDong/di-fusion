#include <torch/extension.h>

std::vector<torch::Tensor> marching_cubes_sparse_interp_cuda(
        torch::Tensor indexer,              // (nx, ny, nz) -> data_id
        torch::Tensor valid_blocks,         // (K, )
        torch::Tensor vec_batch_mapping,
        torch::Tensor cube_sdf,             // (M, rx, ry, rz)
        torch::Tensor cube_std,             // (M, rx, ry, rz)
        int max_n_triangles,                // Maximum number of triangle buffer. //需要调吗
        const std::vector<int>& n_xyz,      // [nx, ny, nz]
        float max_std                       // Prune all vertices // 需要调吗
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("marching_cubes_sparse_interp", &marching_cubes_sparse_interp_cuda, "Marching Cubes with Interpolation (CUDA)");
}
