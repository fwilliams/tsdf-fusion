from abc import ABC, abstractmethod

import cupy as cp
import numpy as np
from numba import njit, prange
from skimage import measure


_cupy_integrate_kernel_src = """
extern "C" __global__ void integrate(float * tsdf_vol,
                          float * weight_vol,
                          float * color_vol,
                          float * vol_dim,
                          float * vol_origin,
                          float * cam_intr,
                          float * cam_pose,
                          float * other_params,
                          float * color_im,
                          float * depth_im,
                          float * weight_im) {
    // Get voxel index
    int gpu_loop_idx = (int) other_params[0];
    int max_threads_per_block = blockDim.x;
    int block_idx = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
    int voxel_idx = gpu_loop_idx * gridDim.x * gridDim.y * gridDim.z * max_threads_per_block + 
                    block_idx * max_threads_per_block + threadIdx.x;
    int vol_dim_x = (int) vol_dim[0];
    int vol_dim_y = (int) vol_dim[1];
    int vol_dim_z = (int) vol_dim[2];
    if (voxel_idx >= (vol_dim_x * vol_dim_y * vol_dim_z)) {
        return;
    }
    // Get voxel grid coordinates (note: be careful when casting)
    float voxel_x = floorf(((float)voxel_idx)/((float)(vol_dim_y*vol_dim_z)));
    float voxel_y = floorf(((float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z))/((float)vol_dim_z));
    float voxel_z = (float)(voxel_idx-((int)voxel_x)*vol_dim_y*vol_dim_z-((int)voxel_y)*vol_dim_z);
    
    // Voxel grid coordinates to world coordinates
    float voxel_size = other_params[1];
    float pt_x = vol_origin[0]+voxel_x*voxel_size;
    float pt_y = vol_origin[1]+voxel_y*voxel_size;
    float pt_z = vol_origin[2]+voxel_z*voxel_size;
    
    // World coordinates to camera coordinates
    float tmp_pt_x = pt_x-cam_pose[0*4+3];
    float tmp_pt_y = pt_y-cam_pose[1*4+3];
    float tmp_pt_z = pt_z-cam_pose[2*4+3];
    float cam_pt_x = cam_pose[0*4+0]*tmp_pt_x+cam_pose[1*4+0]*tmp_pt_y+cam_pose[2*4+0]*tmp_pt_z;
    float cam_pt_y = cam_pose[0*4+1]*tmp_pt_x+cam_pose[1*4+1]*tmp_pt_y+cam_pose[2*4+1]*tmp_pt_z;
    float cam_pt_z = cam_pose[0*4+2]*tmp_pt_x+cam_pose[1*4+2]*tmp_pt_y+cam_pose[2*4+2]*tmp_pt_z;
    
    // Camera coordinates to image pixels
    int pixel_x = (int) roundf(cam_intr[0*3+0]*(cam_pt_x/cam_pt_z)+cam_intr[0*3+2]);
    int pixel_y = (int) roundf(cam_intr[1*3+1]*(cam_pt_y/cam_pt_z)+cam_intr[1*3+2]);
    
    // Skip if voxel is outside the view frustum
    int im_h = (int) other_params[2];
    int im_w = (int) other_params[3];
    if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z < 0) {
        return;
    }
    
    // Skip if voxel corresponds to invalid depth because of ray miss
    float depth_value = depth_im[pixel_y*im_w+pixel_x];
    if (!isfinite(depth_value)) {
        return;
    }
    
    // Allow weighting per-pixel observations (e.g. using cosine of angle between normal and view direction)
    float obs_weight = weight_im[pixel_y*im_w+pixel_x];
    
    // Integrate TSDF
    float trunc_margin = other_params[4];
    float depth_diff = depth_value - cam_pt_z;
    if (depth_diff < -trunc_margin) {
        return;
    }
    float dist = fmin(1.0f,depth_diff/trunc_margin);
    float w_old = weight_vol[voxel_idx];
    // float obs_weight = other_params[5];
    float w_new = w_old + obs_weight;
    weight_vol[voxel_idx] = w_new;
    float old_tsdf = tsdf_vol[voxel_idx];
    if (!isfinite(old_tsdf)) {
        old_tsdf = 1.0;
    }
    tsdf_vol[voxel_idx] = (old_tsdf * w_old + obs_weight * dist) / w_new;
    
    // Integrate color
    float old_color = color_vol[voxel_idx];
    float old_b = floorf(old_color/(256*256));
    float old_g = floorf((old_color-old_b*256*256)/256);
    float old_r = old_color-old_b*256*256-old_g*256;
    float new_color = color_im[pixel_y*im_w+pixel_x];
    float new_b = floorf(new_color/(256*256));
    float new_g = floorf((new_color-new_b*256*256)/256);
    float new_r = new_color-new_b*256*256-new_g*256;
    new_b = fmin(roundf((old_b*w_old+obs_weight*new_b)/w_new),255.0f);
    new_g = fmin(roundf((old_g*w_old+obs_weight*new_g)/w_new),255.0f);
    new_r = fmin(roundf((old_r*w_old+obs_weight*new_r)/w_new),255.0f);
    color_vol[voxel_idx] = new_b*256*256+new_g*256+new_r;
}
"""
_cupy_integrate_kernel = cp.RawKernel(_cupy_integrate_kernel_src, 'integrate')


class TSDVVolumeBase(ABC):
    def __init__(self, vol_bounds, voxel_size, truncation_voxel_distance=5.0):
        """
        Base class for a voxel grid used to accumulate a Truncated Signed Distance Field (TSDF) from
        depth maps.
        :param vol_bounds: A 3x2 shaped array indicating the min/max ranges in voxels of the volume along each axis.
                           i.e. [[min_x, max_x], [min_y, max_y], [min_z, max_z]] specify the integer min and max
                                voxel indices along the x, y, and, z axes respectively.
        :param voxel_size: A floating point number indicating the size of a voxel along an axis (all voxels are cubes)
        :param truncation_voxel_distance: The truncation distance (in voxels) for the TSDF.
        """
        vol_bounds = np.asarray(vol_bounds)
        assert vol_bounds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

        # Define voxel volume parameters
        self._vol_bnds = vol_bounds
        self._voxel_size = float(voxel_size)
        self._trunc_margin = truncation_voxel_distance * self._voxel_size  # truncation on SDF
        self._color_const = 256 * 256

        # Adjust volume bounds and ensure C-order contiguous
        self._vol_dim = np.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size).copy(
            order='C').astype(int)
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(order='C').astype(np.float32)

        self._color_const = 256 ** 2

    @property
    def truncation_margin(self):
        """
        Get the truncation distance (in world units) of the TSDF.
        :return: The truncation distance (in world units) of the TSDF.
        """
        return self._trunc_margin

    @property
    def volume_origin(self):
        """
        Get the origin (bottom, back, left) world coordinate of the TSDF grid.
        :return: The origin (bottom, back, left) world coordinate of the TSDF grid.
        """
        return self._vol_origin

    @property
    def volume_bounds(self):
        """
        Get the bounds (in world coordinates) of the TSDF grid. i.e. [[min_x, max_x], [min_y, max_y], [min_z, max_z]].
        :return: The bounds (in world coordinates) of the TSDF grid. i.e. [[min_x, max_x], [min_y, max_y], [min_z, max_z]].
        """
        return self._vol_bnds

    @property
    def voxel_size(self):
        """
        Get the length of a voxel in the TSDF grid along an axis.
        :return: The length of a voxel in the TSDF grid along an axis.
        """
        return self._voxel_size

    @property
    def grid_coordinates(self):
        """
        Get the center coordinates of each voxel in the TSDF grid as a numpy array of shape [W, H, D, 3] where
            W = width (along x-axis) of the TSDF grid
            H = height (along y-axis) of the TSDF grid
            Z = depth (along z-axis) of the TSDF grid
        :return: The center coordinates of each voxel in the TSDF grid as a numpy array of shape [W, H, D, 3].
        """
        x, y, z = np.mgrid[self.volume_bounds[0, 0]:self.volume_bounds[0, 1]:self.volume_resolution[0]*1j,
                           self.volume_bounds[1, 0]:self.volume_bounds[1, 1]:self.volume_resolution[1]*1j,
                           self.volume_bounds[2, 0]:self.volume_bounds[2, 1]:self.volume_resolution[2]*1j]
        return np.stack([x, y, z], axis=-1)

    @property
    def volume_resolution(self):
        """
        Get the dimensions of the TSDF grid (in voxels) as an array [W, H, D], where
            W = width (along x-axis) of the TSDF grid
            H = height (along y-axis) of the TSDF grid
            Z = depth (along z-axis) of the TSDF grid
        :return: The dimensions of the TSDF grid (in voxels) as an array [W, H, D]
        """
        return self._vol_dim

    @abstractmethod
    def integrate(self, color_im, depth_im, cam_intr, cam_pose, obs_weight=1.):
        pass

    @abstractmethod
    def get_volume(self):
        pass

    def get_point_cloud(self):
        """
        Return a set of points ([N, 3]-shaped array) which lie on the surface of the TSDF represented by this grid.
        Each point lies at the center of a voxel lying on the zero level set of the TSDF.
        """
        tsdf_vol, color_vol = self.get_volume()

        # Marching cubes
        verts = measure.marching_cubes(tsdf_vol, level=0)[0]
        verts_ind = np.round(verts).astype(int)
        verts = verts * self._voxel_size + self._vol_origin

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:, 0], verts_ind[:, 1], verts_ind[:, 2]]
        colors_b = np.floor(rgb_vals / self._color_const)
        colors_g = np.floor((rgb_vals - colors_b*self._color_const) / 256)
        colors_r = rgb_vals - colors_b*self._color_const - colors_g*256
        colors = np.floor(np.asarray([colors_r, colors_g, colors_b])).T
        colors = colors.astype(np.uint8)

        pc = np.hstack([verts, colors])
        return pc

    def get_mesh(self):
        """
        Extract a mesh from the TSDF grid using marching cubes.
        """
        tsdf_vol, color_vol = self.get_volume()

        # Marching cubes
        verts, faces, norms, vals = measure.marching_cubes(tsdf_vol, level=0)
        verts_ind = np.round(verts).astype(int)
        verts = verts*self._voxel_size+self._vol_origin  # voxel grid coordinates to world coordinates

        # Get vertex colors
        rgb_vals = color_vol[verts_ind[:,0], verts_ind[:,1], verts_ind[:,2]]
        colors_b = np.floor(rgb_vals/self._color_const)
        colors_g = np.floor((rgb_vals-colors_b*self._color_const)/256)
        colors_r = rgb_vals-colors_b*self._color_const-colors_g*256
        colors = np.floor(np.asarray([colors_r,colors_g,colors_b])).T
        colors = colors.astype(np.uint8)
        return verts, faces, norms, colors


class TSDFVolumeCPU(TSDVVolumeBase):
    def __init__(self, vol_bounds, voxel_size, truncation_voxel_distance=5.0):
        """
        A voxel grid stored on the CPU used to accumulate a Truncated Signed Distance Field (TSDF) from
        depth maps.

        See TSDFVolumeGPU for a GPU version of this class.

        :param vol_bounds: A 3x2 shaped array indicating the min/max ranges in voxels of the volume along each axis.
                           i.e. [[min_x, max_x], [min_y, max_y], [min_z, max_z]] specify the integer min and max
                                voxel indices along the x, y, and, z axes respectively.
        :param voxel_size: A floating point number indicating the size of a voxel along an axis (all voxels are cubes)
        :param truncation_voxel_distance: The truncation distance (in voxels) for the TSDF.
        """
        super().__init__(vol_bounds, voxel_size, truncation_voxel_distance)

        # Initialize pointers to voxel volume in CPU memory
        self._tsdf_vol_cpu = np.full(self._vol_dim, np.inf).astype(np.float32)
        # for computing the cumulative moving average of observations per voxel
        self._weight_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)
        self._color_vol_cpu = np.zeros(self._vol_dim).astype(np.float32)

        # Get voxel grid coordinates
        xv, yv, zv = np.meshgrid(
            range(self._vol_dim[0]),
            range(self._vol_dim[1]),
            range(self._vol_dim[2]),
            indexing='ij'
        )
        self.vox_coords = np.concatenate([
            xv.reshape(1, -1),
            yv.reshape(1, -1),
            zv.reshape(1, -1)
        ], axis=0).astype(int).T

    @staticmethod
    @njit(parallel=True)
    def _vox2world(vol_origin, vox_coords, vox_size):
        """
        Convert voxel grid coordinates to world coordinates.
        """
        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        cam_pts = np.empty_like(vox_coords, dtype=np.float32)
        for i in prange(vox_coords.shape[0]):
            for j in range(3):
                cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
        return cam_pts

    @staticmethod
    @njit(parallel=True)
    def _cam2pix(cam_pts, intr):
        """
        Convert camera coordinates to pixel coordinates.
        """
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
        for i in prange(cam_pts.shape[0]):
            pix[i, 0] = int(np.round((cam_pts[i, 0] * fx / cam_pts[i, 2]) + cx))
            pix[i, 1] = int(np.round((cam_pts[i, 1] * fy / cam_pts[i, 2]) + cy))
        return pix

    @staticmethod
    @njit(parallel=True)
    def _integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
        """
        Integrate the TSDF volume.
        """
        tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
        w_new = np.empty_like(w_old, dtype=np.float32)
        for i in prange(len(tsdf_vol)):
            w_new[i] = w_old[i] + obs_weight
            old_tsdf_i = tsdf_vol[i] if np.isfinite(tsdf_vol[i]) else 1.0
            tsdf_vol_int[i] = (w_old[i] * old_tsdf_i + obs_weight * dist[i]) / w_new[i]
        return tsdf_vol_int, w_new

    @staticmethod
    def _rigid_transform(xyz, transform):
        """
        Applies a rigid transform to an (N, 3) pointcloud.
        """
        xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
        xyz_t_h = np.dot(transform, xyz_h.T).T
        return xyz_t_h[:, :3]

    def integrate(self, color_im, depth_im, cam_intrinsics, cam_pose, obs_weight=1.):
        """
        Integrate an RGB-D frame into the TSDF volume.

        :param color_im: An RGB image array of shape (H, W, 3).
        :param depth_im: A depth image of shape (H, W).
        :param cam_intrinsics: The camera intrinsics matrix of shape (3, 3).
        :param cam_pose: The camera pose (i.e. extrinsics) of shape (4, 4).
        :param obs_weight: The scalar weight to assign for the current observation.
        :return: None
        """
        im_h, im_w = depth_im.shape

        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32)
        color_im = np.floor(color_im[..., 2]*self._color_const + color_im[..., 1]*256 + color_im[..., 0])

        # Convert voxel grid coordinates to pixel coordinates
        cam_pts = self._vox2world(self._vol_origin, self.vox_coords, self._voxel_size)
        cam_pts = self._rigid_transform(cam_pts, np.linalg.inv(cam_pose))
        pix_z = cam_pts[:, 2]
        pix = self._cam2pix(cam_pts, cam_intrinsics)
        pix_x, pix_y = pix[:, 0], pix[:, 1]

        # Eliminate pixels outside view frustum
        valid_pix = np.logical_and(pix_x >= 0,
                                   np.logical_and(pix_x < im_w,
                                                  np.logical_and(pix_y >= 0,
                                                                 np.logical_and(pix_y < im_h,
                                                                                pix_z > 0))))
        depth_val = np.zeros(pix_x.shape)
        depth_val[valid_pix] = depth_im[pix_y[valid_pix], pix_x[valid_pix]]

        # Integrate TSDF
        depth_diff = depth_val - pix_z
        valid_pts = np.logical_and(
            np.logical_and(depth_val > 0, depth_diff >= -self._trunc_margin),
            np.isfinite(depth_val))
        dist = np.minimum(1, depth_diff / self._trunc_margin)
        valid_vox_x = self.vox_coords[valid_pts, 0]
        valid_vox_y = self.vox_coords[valid_pts, 1]
        valid_vox_z = self.vox_coords[valid_pts, 2]
        w_old = self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
        tsdf_vals = self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
        valid_dist = dist[valid_pts]
        tsdf_vol_new, w_new = self._integrate_tsdf(tsdf_vals, valid_dist, w_old, obs_weight)
        self._weight_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
        self._tsdf_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

        # Integrate color
        old_color = self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z]
        old_b = np.floor(old_color / self._color_const)
        old_g = np.floor((old_color-old_b*self._color_const)/256)
        old_r = old_color - old_b*self._color_const - old_g*256
        new_color = color_im[pix_y[valid_pts], pix_x[valid_pts]]
        new_b = np.floor(new_color / self._color_const)
        new_g = np.floor((new_color - new_b*self._color_const) / 256)
        new_r = new_color - new_b*self._color_const - new_g*256
        new_b = np.minimum(255., np.round((w_old*old_b + obs_weight*new_b) / w_new))
        new_g = np.minimum(255., np.round((w_old*old_g + obs_weight*new_g) / w_new))
        new_r = np.minimum(255., np.round((w_old*old_r + obs_weight*new_r) / w_new))
        self._color_vol_cpu[valid_vox_x, valid_vox_y, valid_vox_z] = new_b*self._color_const + new_g*256 + new_r

    def get_volume(self):
        """
        Get the accumulated TSDF and color volumes.
        :return: A pair (tsdf_vol, color_vol) with shape [W, H, D] and [W, H, D, 3] respectively.
        """
        return self._tsdf_vol_cpu, self._color_vol_cpu


class TSDFVolumeGPU(TSDVVolumeBase):
    def __init__(self, vol_bounds, voxel_size, truncation_voxel_distance=5.0):
        """
        A voxel grid stored on the GPU used to accumulate a Truncated Signed Distance Field (TSDF) from
        depth maps.

        See TSDFVolumeCPU for a CPU version of this class.

        :param vol_bounds: A 3x2 shaped array indicating the min/max ranges in voxels of the volume along each axis.
                           i.e. [[min_x, max_x], [min_y, max_y], [min_z, max_z]] specify the integer min and max
                                voxel indices along the x, y, and, z axes respectively.
        :param voxel_size: A floating point number indicating the size of a voxel along an axis (all voxels are cubes)
        :param truncation_voxel_distance: The truncation distance (in voxels) for the TSDF.
        """
        super().__init__(vol_bounds, voxel_size, truncation_voxel_distance)

        self._tsdf_vol_gpu = cp.full(self._vol_dim, np.inf, dtype=cp.float32)
        self._weight_vol_gpu = cp.zeros(self._vol_dim, dtype=cp.float32)
        self._color_vol_gpu = cp.zeros(self._vol_dim, dtype=cp.float32)

        # Determine block/grid size on GPU
        gpu_dev = cp.cuda.Device(0)
        self._max_gpu_threads_per_block = gpu_dev.attributes['MaxThreadsPerBlock']
        max_grid_dim = [gpu_dev.attributes[a] for a in ['MaxGridDim' + ax for ax in ['X', 'Y', 'Z']]]

        n_blocks = int(np.ceil(float(np.prod(self._vol_dim)) / float(self._max_gpu_threads_per_block)))
        grid_dim_x = min(max_grid_dim[0], int(np.floor(np.cbrt(n_blocks))))
        grid_dim_y = min(max_grid_dim[1], int(np.floor(np.sqrt(n_blocks / grid_dim_x))))
        grid_dim_z = min(max_grid_dim[2], int(np.ceil(float(n_blocks) / float(grid_dim_x * grid_dim_y))))
        self._max_gpu_grid_dim = np.array([grid_dim_x, grid_dim_y, grid_dim_z]).astype(int)
        self._n_gpu_loops = int(np.ceil(
            float(np.prod(self._vol_dim)) / float(np.prod(self._max_gpu_grid_dim) * self._max_gpu_threads_per_block)))

    def integrate(self, color_im, depth_im, cam_intrinsics, cam_pose, obs_weight=1.0):
        """
        Integrate an RGB-D frame into the TSDF volume.

        :param color_im: An RGB image array of shape (H, W, 3).
        :param depth_im: A depth image of shape (H, W).
        :param cam_intrinsics: The camera intrinsics matrix of shape (3, 3).
        :param cam_pose: The camera pose (i.e. extrinsics) of shape (4, 4).
        :param obs_weight: The scalar weight to assign for the current observation.
        :return: None
        """
        im_w, im_h = depth_im.shape

        # Fold RGB color image into a single channel image
        color_im = color_im.astype(np.float32)
        color_im = np.floor(color_im[..., 2]*self._color_const + color_im[..., 1]*256 + color_im[..., 0])

        if np.isscalar(obs_weight):
            obs_weight = np.ones_like(depth_im)

        # Copy input arguments to cupy arrays
        vol_dim_cp = cp.asarray(self._vol_dim.astype(np.float32))
        vol_origin_cp = cp.asarray(self._vol_origin.astype(np.float32))
        cam_intrinsics_cp = cp.asarray(cam_intrinsics.reshape(-1).astype(np.float32))
        cam_pose_cp = cp.asarray(cam_pose.reshape(-1).astype(np.float32))
        extra_args_cp = cp.array([
            0,
            self._voxel_size,
            im_h,
            im_w,
            self._trunc_margin,
        ], dtype=cp.float32)
        color_im_cp = cp.asarray(color_im.reshape(-1).astype(np.float32))
        depth_im_cp = cp.asarray(depth_im.reshape(-1).astype(np.float32))
        weight_im_cp = cp.asarray(obs_weight.reshape(-1).astype(np.float32))

        for gpu_loop_idx in range(self._n_gpu_loops):
            threads_per_block = (self._max_gpu_threads_per_block, 1, 1)
            blocks_per_grid = tuple([x for x in self._max_gpu_grid_dim])
            extra_args_cp[0] = gpu_loop_idx
            _cupy_integrate_kernel(
                blocks_per_grid, threads_per_block,
                args=(self._tsdf_vol_gpu,
                      self._weight_vol_gpu,
                      self._color_vol_gpu,
                      vol_dim_cp,
                      vol_origin_cp,
                      cam_intrinsics_cp,
                      cam_pose_cp,
                      extra_args_cp,
                      color_im_cp,
                      depth_im_cp,
                      weight_im_cp)
            )
        cp.cuda.stream.get_current_stream().synchronize()

    def get_volume(self):
        """
        Get the accumulated TSDF and color volumes.
        :return: A pair (tsdf_vol, color_vol) with shape [W, H, D] and [W, H, D, 3] respectively.
        """
        return cp.asnumpy(self._tsdf_vol_gpu), cp.asnumpy(self._color_vol_gpu)

