import numpy as np
import cv2


def _update_extrinsics(
        extrinsics, 
        angle, 
        trans=None, 
        rotate_axis='y'):
    r""" Uptate camera extrinsics when rotating it around a standard axis.

    Args:
        - extrinsics: Array (3, 3)
        - angle: Float
        - trans: Array (3, )
        - rotate_axis: String

    Returns:
        - Array (3, 3)
    """
    E = extrinsics
    inv_E = np.linalg.inv(E)

    camrot = inv_E[:3, :3]
    campos = inv_E[:3, 3]
    if trans is not None:
        campos -= trans

    rot_y_axis = camrot.T[1, 1]
    if rot_y_axis < 0.:
        angle = -angle
    
    rotate_coord = {
        'x': 0, 'y': 1, 'z':2
    }
    grot_vec = np.array([0., 0., 0.])
    grot_vec[rotate_coord[rotate_axis]] = angle
    grot_mtx = cv2.Rodrigues(grot_vec)[0].astype('float32')

    rot_campos = grot_mtx.dot(campos) 
    rot_camrot = grot_mtx.dot(camrot)
    if trans is not None:
        rot_campos += trans
    
    new_E = np.identity(4)
    new_E[:3, :3] = rot_camrot.T
    new_E[:3, 3] = -rot_camrot.T.dot(rot_campos)

    return new_E

    
def _get_camrot(campos, lookat=None, inv_camera=False):
    r""" Compute rotation part of extrinsic matrix from camera posistion and
         where it looks at.

    Args:
        - campos: Array (3, )
        - lookat: Array (3, )
        - inv_camera: Boolean

    Returns:
        - Array (3, 3)

    Reference: http://ksimek.github.io/2012/08/22/extrinsic/
    """

    if lookat is None:
        lookat = np.array([0., 0., 0.], dtype=np.float32)

    # define up, forward, and right vectors
    up = np.array([0., 1., 0.], dtype=np.float32)
    if inv_camera:
        up[1] *= -1.0
    forward = lookat - campos
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    up /= np.linalg.norm(up)

    camrot = np.array([right, up, forward], dtype=np.float32)
    return camrot


def tpose_camera(img_size, radius, focal, angle):
    x = 0.
    y = -0.1
    z = radius
    R = cv2.Rodrigues(np.array([0, angle, 0], dtype='float32'))[0]
    campos = R @ np.array([x, y, z], dtype='float32')
    camrot = _get_camrot(campos, 
                        lookat=np.array([0, y, 0.]),
                        inv_camera=True)

    E = np.eye(4, dtype='float32')
    E[:3, :3] = camrot
    E[:3, 3] = -camrot.dot(campos)

    K = np.eye(3, dtype='float32')
    K[0, 0] = focal
    K[1, 1] = focal
    K[:2, 2] = img_size / 2.

    return K, E


def rotate_camera_by_frame_idx(
        extrinsics, 
        frame_idx, 
        trans=None,
        rotate_axis='y',
        period=196,
        inv_angle=False):
    r""" Get camera extrinsics based on frame index and rotation period.

    Args:
        - extrinsics: Array (3, 3)
        - frame_idx: Integer
        - trans: Array (3, )
        - rotate_axis: String
        - period: Integer
        - inv_angle: Boolean (clockwise/counterclockwise)

    Returns:
        - Array (3, 3)
    """

    angle = 2 * np.pi * (frame_idx / period)
    if inv_angle:
        angle = -angle
    return _update_extrinsics(
                extrinsics, angle, trans, rotate_axis)


def apply_global_tfm_to_camera(E, Rh, Th):
    r""" Get camera extrinsics that considers global transformation.

    Args:
        - E: Array (3, 3)
        - Rh: Array (3, )
        - Th: Array (3, )
        
    Returns:
        - Array (3, 3)
    """

    global_tfms = np.eye(4)  #(4, 4)
    global_rot = cv2.Rodrigues(Rh)[0].T
    global_trans = Th
    global_tfms[:3, :3] = global_rot
    global_tfms[:3, 3] = -global_rot.dot(global_trans)
    # _global_tfms = np.eye(4)  #(4, 4)
    # _global_rot = cv2.Rodrigues(Rh)[0]
    # _global_tfms[:3, :3] = _global_rot
    # _global_tfms[:3, 3] = Th
    return E.dot(np.linalg.inv(global_tfms))


def get_rays_from_KRT(H, W, K, R, T):
    r""" Sample rays on an image based on camera matrices (K, R and T)

    Args:
        - H: Integer
        - W: Integer
        - K: Array (3, 3)
        - R: Array (3, 3)
        - T: Array (3, )
        
    Returns:
        - rays_o: Array (H, W, 3)
        - rays_d: Array (H, W, 3)
    """

    # calculate the camera origin
    #rays_o = -np.dot(R.T, T).ravel()
    rays_o = -np.dot(np.linalg.inv(R), T).ravel()
    
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    return rays_o, rays_d


def project_3d_points_to_camera_plane(X, K, E):
    R = E[:3, :3]
    T = E[:3, 3]
    xyz = (X @ R.T + T) @ K.T
    xy = xyz[:, :2] / xyz[:, 2:]
    return xy


def rays_intersect_3d_bbox(bounds, ray_o, ray_d):
    r"""calculate intersections with 3d bounding box
        Args:
            - bounds: dictionary or list
            - ray_o: (N_rays, 3)
            - ray_d, (N_rays, 3)
        Output:
            - near: (N_VALID_RAYS, )
            - far: (N_VALID_RAYS, )
            - mask_at_box: (N_RAYS, )
    """

    if isinstance(bounds, dict):
        bounds = np.stack([bounds['min_xyz'], bounds['max_xyz']], axis=0)
    assert bounds.shape == (2,3)

    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    nominator = bounds[None] - ray_o[:, None] # (N_rays, 2, 3)
    # calculate the step of intersections at six planes of the 3d bounding box
    ray_d[np.abs(ray_d) < 1e-5] = 1e-5
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6) # (N_rays, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None] # (N_rays, 6, 3)
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))  # (N_rays, 6)
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2  #(N_rays, )
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3) # (N_VALID_rays, 2, 3)

    # calculate the step of intersections
    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o, axis=1) / norm_ray
    near = np.minimum(d0, d1)
    far = np.maximum(d0, d1)

    return near, far, mask_at_box


def project_3d_bbox(bbox, rays_o, rays_d, H, W):
    _, _, ray_mask = rays_intersect_3d_bbox(bbox, rays_o, rays_d)
    mask_rows, mask_cols = np.nonzero(ray_mask.reshape(H, W))
    row_from = int(np.min(mask_rows))
    row_to = int(np.max(mask_rows) + 1)
    col_from = int(np.min(mask_cols))
    col_to = int(np.max(mask_cols) + 1)
    bbox_2d = [col_from, row_from, col_to, row_to]
    return np.array(bbox_2d)


def cam_opencv_to_opengl(R, T):
    # see: https://stackoverflow.com/questions/44375149/opencv-to-opengl-coordinate-system-transform
    # R_cv_to_gl(R @ x + T)
    R_cv_to_gl = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0].astype('float32')
    R_new = R_cv_to_gl @ R
    T_new = R_cv_to_gl @ T
    return R_new, T_new


def get_render(mesh, H, W, K, E):
    import pyrender
    render_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(render_mesh)

    yfov = 2 * np.arctan2(H, 2 * K[1, 1])
    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=W/H)
    camera_pose = np.eye(4)
    R = E[:3, :3]
    T = E[:3, 3]
    R, T = cam_opencv_to_opengl(R, T)
    camera_pose[:3, :3] = R
    camera_pose[:3, 3] = T
    scene.add(camera, pose=np.linalg.inv(camera_pose))
    
    dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(dl, pose=camera_pose)
    
    r = pyrender.OffscreenRenderer(W, H)
    color, _ = r.render(scene)
    return color
