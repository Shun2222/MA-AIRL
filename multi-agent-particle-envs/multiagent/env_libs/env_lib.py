import numpy as np 

def pos_to_row_col(grid_size, pos):
    one_grid_size = np.array([2/grid_size[0], 2/grid_size[1]])
    row_col = pos + np.ones(2)
    row_col = row_col//one_grid_size
    row_col = np.array([ grid_size[0]-row_col[1], row_col[0]])
    return row_col

def row_col_to_pos(grid_size, row_col):
    one_grid_size = np.array([2/grid_size[0], 2/grid_size[1]])
    pos = row_col * one_grid_size + one_grid_size*0.5 
    pos -= np.ones(2)
    pos = np.array([pos[1], -pos[0]])
    return pos
    
