import numpy as np

def clean_field(field1, field2):
    # Use NumPy for efficient boolean operations
    field1_array = np.array(field1, dtype=np.int8)
    field2_array = np.array(field2, dtype=np.int8)
    return (field1_array & field2_array).tolist()

def fake_place(x, y, coordinates, field):
    field_copy = np.array(field, dtype=np.int8)
    
    # Convert coordinates to array indices for vectorized placement
    coords_x = [x + c[0] for c in coordinates]
    coords_y = [y + c[1] for c in coordinates]
    
    # Check boundaries in one step
    if (min(coords_x) < 0 or max(coords_x) >= 10 or 
        min(coords_y) < 0 or max(coords_y) >= 10):
        return 0, field
    
    # Check if placement is valid (no overlaps)
    if any(field_copy[coords_y[i]][coords_x[i]] == 1 for i in range(len(coordinates))):
        return 0, field
    
    # Place the piece
    for i in range(len(coordinates)):
        field_copy[coords_y[i]][coords_x[i]] = 1
    
    # Check for completed lines and columns
    lines, columns = [], []
    
    # Check lines (rows)
    for line in range(10):
        if np.all(field_copy[line] == 1):
            lines.append(line)
            field_copy[line] = 0
    
    # Check columns
    for col in range(10):
        if np.all(field_copy[:, col] == 1):
            columns.append(col)
            field_copy[:, col] = 0
    
    # Calculate score
    score = len(coordinates) + len(lines) * 10 + len(columns) * 10
    
    return score, field_copy.tolist()

def check_fake_lines(field):
    field_np = np.array(field)
    lines = []
    
    for line in range(10):
        if np.all(field_np[line] == 1):
            lines.append(line)
            field_np[line] = 0
    
    return lines, field_np.tolist()

def check_fake_columns(field):
    field_np = np.array(field)
    columns = []
    
    for col in range(10):
        if np.all(field_np[:, col] == 1):
            columns.append(col)
            field_np[:, col] = 0
    
    return columns, field_np.tolist()

def set_fake_filled(x, y, full, fake_field):
    fake_field[y][x] = full
    return fake_field

def fake_fits(x, y, coordinates, field):
    # Check boundaries first
    for dx, dy in coordinates:
        tx, ty = x + dx, y + dy
        if not (0 <= tx < 10 and 0 <= ty < 10):
            return False
        if field[ty][tx] == 1:
            return False
    return True

def valid_moves_for_block_on_field(block, field):
    moves = []
    coords = block.coord_array
    
    # Determine the minimal search space based on block size
    min_x, max_x = min(c[0] for c in coords), max(c[0] for c in coords)
    min_y, max_y = min(c[1] for c in coords), max(c[1] for c in coords)
    
    # Only search positions where the block can fit
    for y in range(-min_y, 10 - max_y):
        for x in range(-min_x, 10 - max_x):
            if fake_fits(x, y, coords, field):
                moves.append([x, y, block])
    
    return moves
