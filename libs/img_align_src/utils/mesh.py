import cv2
import numpy as np


def grid2mesh(grid_ridx, grid_cidx, meshgrid_x, meshgrid_y):
    tl_pt = [meshgrid_x[grid_ridx, grid_cidx], meshgrid_y[grid_ridx, grid_cidx]]
    tr_pt = [meshgrid_x[grid_ridx, grid_cidx + 1], meshgrid_y[grid_ridx, grid_cidx + 1]]
    br_pt = [meshgrid_x[grid_ridx + 1, grid_cidx + 1], meshgrid_y[grid_ridx + 1, grid_cidx + 1]]
    bl_pt = [meshgrid_x[grid_ridx + 1, grid_cidx], meshgrid_y[grid_ridx + 1, grid_cidx]]
    
    vertex_pts = [tl_pt, tr_pt, br_pt, bl_pt]
    vertex_mesh_idxes = [
        (grid_ridx, grid_cidx),
        (grid_ridx, grid_cidx + 1),
        (grid_ridx + 1, grid_cidx + 1),
        (grid_ridx + 1, grid_cidx)
    ]
    return vertex_mesh_idxes, vertex_pts


def pt_in_grid(pt, vertex_pts):
    vertex_pts_arr = np.array(vertex_pts, dtype= np.int32)
    test_res = cv2.pointPolygonTest(vertex_pts_arr, pt, measureDist= False)
    in_grid = test_res >= 0
    return in_grid


def transform_pt(src_pt, src_meshgrid_xy, tgt_meshgrid_xy):
    src_meshgrid_x, src_meshgrid_y = src_meshgrid_xy
    tgt_meshgrid_x, tgt_meshgrid_y = tgt_meshgrid_xy

    mesh_rsize, mesh_csize = tgt_meshgrid_x.shape
    grid_rsize, grid_csize = mesh_rsize - 1, mesh_csize - 1

    in_gridx = None
    in_gcidx = None
    in_src_vertex_pts = []
    in_tgt_vertex_pts = []
    for gridx in range(grid_rsize):
        for gcidx in range(grid_csize):
            _, src_vertex_pts = grid2mesh(gridx, gcidx, src_meshgrid_x, src_meshgrid_y)
            _, tgt_vertex_pts = grid2mesh(gridx, gcidx, tgt_meshgrid_x, tgt_meshgrid_y)

            is_in_grid = pt_in_grid(src_pt, src_vertex_pts)
            if is_in_grid:
                in_gridx = gridx
                in_gcidx = gcidx
                in_src_vertex_pts = src_vertex_pts
                in_tgt_vertex_pts = tgt_vertex_pts
                break

        if in_gridx is not None:
            break

    tgt_pt = () 
    if in_gridx is None:
        return tgt_pt

    in_src_vertex_pts_arr = np.array(in_src_vertex_pts, dtype = np.float32)
    in_tgt_vertex_pts_arr = np.array(in_tgt_vertex_pts, dtype = np.float32)

    local_H = cv2.getPerspectiveTransform(in_src_vertex_pts_arr, in_tgt_vertex_pts_arr)
    src_pt_arr = np.array(src_pt, dtype = np.float32).reshape(-1, 1, 2)
    tgt_pt_arr = cv2.perspectiveTransform(src_pt_arr, local_H)
    tgt_pt = tuple(tgt_pt_arr.tolist()[0][0])

    return tgt_pt

