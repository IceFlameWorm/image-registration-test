import cv2


def draw_grid(img_bgr, grid, color = [255,0,0], thickness = 1):
    res_img = img_bgr
    tl, tr, br, bl = grid
    tl = tuple(tl)
    tr = tuple(tr)
    br = tuple(br)
    bl = tuple(bl)
    res_img = cv2.line(res_img, tl, tr, color, thickness)
    res_img = cv2.line(res_img, tr, br, color, thickness)
    res_img = cv2.line(res_img, br, bl, color, thickness)
    res_img = cv2.line(res_img, bl, tl, color, thickness)
    return res_img


def draw_meshgrid(img_bgr, meshgrid_xy, color = [255,0,0], thickness = 1):
    meshgrid_x, meshgrid_y = meshgrid_xy
    mesh_h, mesh_w = meshgrid_x.shape
    grids = []
    res_img = img_bgr
    
    for hidx in range(0, mesh_h - 1):
        for widx in range(0, mesh_w - 1):
            grid_tl = [int(meshgrid_x[hidx][widx]), int(meshgrid_y[hidx][widx])]
            grid_tr = [int(meshgrid_x[hidx][widx + 1]), int(meshgrid_y[hidx][widx + 1])]
            grid_br = [int(meshgrid_x[hidx + 1][widx + 1]), int(meshgrid_y[hidx + 1][widx + 1])]
            grid_bl = [int(meshgrid_x[hidx + 1][widx]), int(meshgrid_y[hidx + 1][widx])]
            
            grid = [grid_tl, grid_tr, grid_br, grid_bl]
            grids.append(grid)
            res_img = draw_grid(res_img, grid, color)
    
    return res_img, grids