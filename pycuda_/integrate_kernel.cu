__global__ void integrate(float * tsdf_vol,
                                      float * weight_vol,
                                      float * color_vol,
                                      float * vol_dim,
                                      float * vol_origin,
                                      float * cam_intr,
                                      float * cam_pose,
                                      float * other_params,
                                      float * color_im,
                                      float * depth_im) {
  // Get voxel index
  int gpu_loop_idx = (int) other_params[0];
  int max_threads_per_block = blockDim.x;
  int block_idx = blockIdx.z*gridDim.y*gridDim.x+blockIdx.y*gridDim.x+blockIdx.x;
  int voxel_idx = gpu_loop_idx*gridDim.x*gridDim.y*gridDim.z*max_threads_per_block+block_idx*max_threads_per_block+threadIdx.x;
  int vol_dim_x = (int) vol_dim[0];
  int vol_dim_y = (int) vol_dim[1];
  int vol_dim_z = (int) vol_dim[2];
  if (voxel_idx > vol_dim_x*vol_dim_y*vol_dim_z)
      return;
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
  // Skip if outside view frustum
  int im_h = (int) other_params[2];
  int im_w = (int) other_params[3];
  if (pixel_x < 0 || pixel_x >= im_w || pixel_y < 0 || pixel_y >= im_h || cam_pt_z<0)
      return;
  // Skip invalid depth
  float depth_value = depth_im[pixel_y*im_w+pixel_x];
  if (depth_value == 0)
      return;
  // Integrate TSDF
  float trunc_margin = other_params[4];
  float depth_diff = depth_value-cam_pt_z;
  if (depth_diff < -trunc_margin)
      return;
  float dist = fmin(1.0f,depth_diff/trunc_margin);
  float w_old = weight_vol[voxel_idx];
  float obs_weight = other_params[5];
  float w_new = w_old + obs_weight;
  weight_vol[voxel_idx] = w_new;
  tsdf_vol[voxel_idx] = (tsdf_vol[voxel_idx]*w_old+obs_weight*dist)/w_new;
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