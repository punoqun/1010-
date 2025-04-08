import random
import math
import time
from itertools import permutations
from blocks import SimpleBlock
from utils import fake_place, fake_fits, valid_moves_for_block_on_field

class AI:
    def __init__(self, AItype):
        self.AItype = AItype
        self.next_moves = [None, None, None]
        self.current_move = None

    def play_once(self, fake_main):
        # Store current blocks to check if the move is valid after computation
        current_blocks_before = fake_main.game.current_blocks.copy()
        
        if self.AItype == 'greedy':
            self.current_move = self.get_greedy_move(fake_main=fake_main)
        elif self.AItype == 'MCTS':
            ai = self.MCTSAI(_main=fake_main)
            self.current_move = ai.get_MCTS_move()
        elif self.AItype == 'heuristic':
            self.current_move = self.get_heuristic_move(fake_main=fake_main)
            
        # Validate the move: ensure the selected block is still available and usable
        if self.current_move is not None:
            _, _, block = self.current_move
            if block not in fake_main.game.current_blocks or getattr(block, 'is_used', False):
                # Reset properties and recalculate
                for b in fake_main.game.current_blocks:
                    if hasattr(b, 'is_used'):
                        delattr(b, 'is_used')
                self.play_once(fake_main)

    @staticmethod
    def get_greedy_move(fake_main):
        # Only consider valid and unused blocks
        moves = fake_main.game.valid_moves()  # This now excludes used blocks
        if not moves:
            return None
            
        random.shuffle(moves)
        best_move = moves[0]  # Default to first move
        best_score = 0
        for move_ in moves:
            field_copy = [row[:] for row in fake_main.game.copy_field()]
            score, _ = fake_place(x=move_[0], y=move_[1], coordinates=move_[2].coord_array, field=field_copy)
            if score > best_score:
                best_score = score
                best_move = move_
        return best_move

    def get_heuristic_moves(self, fake_main):
        current_blocks = fake_main.game.current_blocks
        perms = list(permutations(current_blocks))
        best_score = 0
        best_move_perm = None
        for perm in perms:
            field_copy = [row[:] for row in fake_main.game.copy_field()]
            temp_score = 0
            move_list_of_perm = []
            for block in perm:
                temp_moves = valid_moves_for_block_on_field(block, field_copy)
                if len(temp_moves) == 0:
                    break
                block_score = -1
                block_move = None
                block_field = None
                random.shuffle(temp_moves)
                for move_ in temp_moves:
                    _, field = fake_place(x=move_[0], y=move_[1], coordinates=move_[2].coord_array, 
                                         field=[row[:] for row in field_copy])
                    heu_score = self.get_heuristic_score(field)
                    if heu_score+_ > block_score:
                        block_score = heu_score+_
                        block_move = move_
                        block_field = [row[:] for row in field]
                move_list_of_perm.append(block_move)
                temp_score += block_score
                field_copy = [row[:] for row in block_field]
            if temp_score > best_score:
                best_score = temp_score
                best_move_perm = move_list_of_perm
        self.next_moves = best_move_perm

    @staticmethod
    def get_heuristic_score(field):
        score = 0
        rows = len(field)
        cols = len(field[0]) if rows > 0 else 0
        # Define neighbor offsets along with bonus: (dy, dx, bonus)
        neighbor_offsets = [
            (-1,  0, 2),  # Up
            (-1, -1, 4),  # Up-left
            (-1,  1, 4),  # Up-right
            ( 1,  0, 2),  # Down
            ( 1, -1, 4),  # Down-left
            ( 1,  1, 4),  # Down-right
            ( 0, -1, 2),  # Left
            ( 0,  1, 2)   # Right
        ]
        for y in range(rows):
            for x in range(cols):
                if field[y][x] == 0:
                    # Base score for an empty cell
                    score += 1
                    # Check all valid neighbors
                    for dy, dx, bonus in neighbor_offsets:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx < cols and field[ny][nx] == 0:
                            score += bonus
        return score

    def get_heuristic_move(self, fake_main):
        if self.next_moves[0] is not None:
            move = self.next_moves[0]
            self.next_moves[0] = None
            return move
        if self.next_moves[1] is not None:
            move = self.next_moves[1]
            self.next_moves[1] = None
            return move
        if self.next_moves[2] is not None:
            move = self.next_moves[2]
            self.next_moves[2] = None
            return move
        self.get_heuristic_moves(fake_main)
        return self.get_heuristic_move(fake_main)

    def get_smart_greedy_move(self):
        """Find a greedy move that focuses on clearing lines and preserving board space"""
        moves = self.main.game.valid_moves()
        if not moves:
            return None
        
        best_move = moves[0]
        best_score = float('-inf')
        
        for move in moves:
            field_copy = [row[:] for row in self.main.game.copy_field()]
            immediate_score, new_field = fake_place(
                move[0], move[1], move[2].coord_array, field_copy
            )
            
            # Calculate a holistic score:
            # 1. Points from immediate placement and line clears
            # 2. Potential for future line clears
            # 3. Open spaces remaining
            # 4. Penalty for creating isolated empty spaces
            
            # Base score from placement
            total_score = immediate_score
            
            # Bonus for immediate line clears (beyond piece placement)
            if immediate_score > len(move[2].coord_array):
                total_score *= 3  # Triple score for line-clearing moves
            
            # Bonus for potential future line clears
            total_score += self._evaluate_potential_lines(new_field)
            
            # Bonus for open spaces
            open_spaces = sum(1 for row in new_field for cell in row if cell == 0)
            total_score += open_spaces * 0.5
            
            # Penalty for isolated empty spaces
            isolated_penalty = self._count_isolated_spaces(new_field)
            total_score -= isolated_penalty * 2
            
            if total_score > best_score:
                best_score = total_score
                best_move = move
        
        return best_move

    def _count_isolated_spaces(self, field):
        """Count isolated empty spaces that are hard to fill"""
        count = 0
        for y in range(10):
            for x in range(10):
                if field[y][x] == 0:
                    # Check if surrounded by filled cells or boundaries
                    surrounded = 0
                    for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                        nx, ny = x + dx, y + dy
                        if nx < 0 or nx >= 10 or ny < 0 or ny >= 10 or field[ny][nx] == 1:
                            surrounded += 1
                    
                    if surrounded >= 3:  # 3 or more sides blocked
                        count += 1
        return count

    class MCTSAI:
        def __init__(self, _main):
            self.main = _main
            self.heuristic = AI.get_heuristic_score
            self.exploration_weight = 1.41
            self.simulation_limit = 30
            self.search_time = 0.3
            self.move_cache = {}
            
            # Dramatically increase line clear weights
            self.line_clear_weight = 50      # Increased from 30
            self.potential_line_weight = 10  # Increased from 3
            self.open_space_weight = 0.8
            self.corner_penalty_weight = 2
            
        def get_MCTS_move(self):
            all_moves = self.main.game.valid_moves()
            if not all_moves:
                return None
            if len(all_moves) == 1:
                return all_moves[0]
            
            # First, check for any immediate line clearing moves
            # This bypasses the full MCTS if we find a good line clearing opportunity
            best_line_clear_move = None
            best_line_clear_score = 0
            
            for move in all_moves:
                score, new_field = fake_place(move[0], move[1], move[2].coord_array, 
                                             self.main.game.copy_field())
                
                piece_size = len(move[2].coord_array)
                lines_cleared = (score - piece_size) / 10 if score > piece_size else 0
                
                # If this move clears lines, consider taking it immediately
                if lines_cleared > 0:
                    # Calculate how attractive this line clear is
                    clear_score = score + (lines_cleared * 20)
                    
                    # Add bonuses for:
                    # 1. Clearing multiple lines at once
                    if lines_cleared > 1:
                        clear_score *= 1.5
                    
                    # 2. Board state after the clear
                    potential_future = self._evaluate_potential_lines(new_field)
                    connected_space = self._evaluate_connected_spaces(new_field)
                    clear_score += potential_future + connected_space
                    
                    if clear_score > best_line_clear_score:
                        best_line_clear_score = clear_score
                        best_line_clear_move = move
            
            # If we found a good line clearing move, take it immediately
            # (Bypass full MCTS for obvious good moves)
            if best_line_clear_move and best_line_clear_score > 30:
                return best_line_clear_move
            
            # Otherwise proceed with regular move scoring
            move_scores = []
            for move in all_moves:
                score, new_field = fake_place(move[0], move[1], move[2].coord_array, 
                                             self.main.game.copy_field())
                
                piece_size = len(move[2].coord_array)
                total_score = score
                
                # Significant bonus for clearing lines
                clears_line = score > piece_size
                if clears_line:
                    lines_cleared = (score - piece_size) / 10
                    # Even higher multiplier for line clears
                    total_score += lines_cleared * self.line_clear_weight * 1.5
                
                # Check potential for future line clears
                potential = self._evaluate_potential_lines(new_field)
                total_score += potential * self.potential_line_weight
                
                # Board layout quality
                connected_space_bonus = self._evaluate_connected_spaces(new_field)
                total_score += connected_space_bonus
                
                # Avoid problematic board states
                corner_penalty = self._count_hard_corners(new_field) * self.corner_penalty_weight
                total_score -= corner_penalty
                
                if piece_size > 3 and self._is_edge_placement(move[0], move[1], move[2].coord_array):
                    total_score += piece_size
                
                move_scores.append((move, total_score, new_field))
            
            # Sort moves by score
            move_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Skip MCTS if there's a clearly superior move
            if move_scores and len(move_scores) > 1 and move_scores[0][1] > move_scores[1][1] * 1.8 and move_scores[0][1] > 35:
                return move_scores[0][0]
            
            # Run MCTS on top moves
            top_moves = move_scores[:min(5, len(move_scores))]
            
            root = MCTSNode(self.main.game.copy_field(), self.main)
            
            # Pre-expand root with top candidate moves
            for move, score, new_field in top_moves:
                child = MCTSNode(new_field, self.main, root)
                child.move_played = move
                child.immediate_score = score
                # Flag line-clearing moves explicitly
                piece_size = len(move[2].coord_array)
                child.cleared_lines = score > piece_size
                root.children.append(child)
                
            # Run MCTS for the allocated time
            start_time = time.time()
            iterations = 0
            while time.time() - start_time < self.search_time and iterations < 250:
                node = self._select(root)
                if node.visits > 0:
                    node = self._expand(node)
                reward = self._simulate(node)
                self._backpropagate(node, reward)
                iterations += 1
            
            # Select best child with explicit preference for line-clearing moves
            best_child = None
            best_score = float('-inf')
            
            for child in root.children:
                if child.visits > 0:
                    # Base value from MCTS
                    avg_value = child.value / child.visits
                    visit_confidence = math.sqrt(child.visits) / 10
                    
                    # Calculate total value with extra bonus for line-clearing moves
                    line_clear_bonus = 15 if getattr(child, 'cleared_lines', False) else 0
                    total_value = avg_value + visit_confidence + line_clear_bonus
                    
                    if total_value > best_score:
                        best_score = total_value
                        best_child = child
            
            if not best_child:
                return move_scores[0][0] if move_scores else all_moves[0]
                
            return best_child.move_played
        
        def _select(self, node):
            # Selection logic with stronger bias for line-clearing nodes
            while node.children:
                unvisited = [child for child in node.children if child.visits == 0]
                if unvisited:
                    # Prioritize unvisited line-clearing nodes
                    line_clearing_nodes = [n for n in unvisited if getattr(n, 'cleared_lines', False)]
                    if line_clearing_nodes:
                        return random.choice(line_clearing_nodes)
                    return random.choice(unvisited)
                
                total_visits = sum(child.visits for child in node.children)
                log_total = math.log(total_visits + 1)
                best_score = float('-inf')
                best_child = None
                
                for child in node.children:
                    exploit = child.value / child.visits
                    explore_weight = self.exploration_weight * (1 - (child.visits / (child.visits + 50)))
                    explore = explore_weight * math.sqrt(log_total / child.visits)
                    
                    # Much larger bonus for line-clearing nodes
                    line_clear_bonus = 2.0 if getattr(child, 'cleared_lines', False) else 0
                    
                    ucb = exploit + explore + line_clear_bonus
                    
                    if ucb > best_score:
                        best_score = ucb
                        best_child = child
                
                node = best_child
            
            return node
            
        def _simulate(self, node):
            # Make a copy of the field for simulation
            sim_field = [row[:] for row in node.field]
            
            # Score starts with immediate move score
            score = getattr(node, 'immediate_score', 0)
            
            # Track available blocks
            available_blocks = [block for block in self.main.game.current_blocks]
            
            # Remove block just placed
            if node.move_played:
                block_to_remove = node.move_played[2]
                for i, block in enumerate(available_blocks):
                    if block is block_to_remove:
                        available_blocks.pop(i)
                        break
            
            # Play moves until game over or limit reached
            steps = 0
            line_clears = 0
            
            while steps < self.simulation_limit and available_blocks:
                sim_moves = []
                for y in range(10):
                    for x in range(10):
                        for block in available_blocks:
                            if fake_fits(x, y, block.coord_array, sim_field):
                                sim_moves.append([x, y, block])
                
                if not sim_moves:
                    break
                
                weighted_moves = []
                for move in sim_moves:
                    move_score, new_field = fake_place(move[0], move[1], move[2].coord_array, 
                                                    [row[:] for row in sim_field])
                    
                    piece_size = len(move[2].coord_array)
                    lines_cleared = (move_score - piece_size) / 10 if move_score > piece_size else 0
                    
                    # Strategic evaluation with heavy line-clear focus
                    strategic_score = move_score
                    
                    # Dramatically boost line-clearing moves
                    if lines_cleared > 0:
                        strategic_score += lines_cleared * self.line_clear_weight * 2
                        # Extra bonus for multiple line clears
                        if lines_cleared > 1:
                            strategic_score *= 1.5
                    
                    # Check for setup for future line clears
                    potential = self._evaluate_potential_lines(new_field)
                    strategic_score += potential * self.potential_line_weight
                    
                    # Value open space and large connected regions
                    open_spaces = sum(1 for row in new_field for cell in row if cell == 0)
                    strategic_score += open_spaces * self.open_space_weight
                    
                    # Avoid creating isolated cells
                    corner_penalty = self._count_hard_corners(new_field)
                    strategic_score -= corner_penalty * self.corner_penalty_weight
                    
                    # Favor large pieces (more efficient use of space)
                    piece_size_bonus = piece_size * 1.5
                    strategic_score += piece_size_bonus
                    
                    weighted_moves.append((move, strategic_score))
                
                # Always favor line-clearing moves with high probability
                line_clearing_moves = [(m, s) for m, s in weighted_moves if 
                                      fake_place(m[0], m[1], m[2].coord_array, [row[:] for row in sim_field])[0] > 
                                      len(m[2].coord_array)]
                
                if line_clearing_moves and random.random() < 0.9:  # 90% chance to pick a line-clearing move if available
                    move = max(line_clearing_moves, key=lambda x: x[1])[0]
                else:
                    # Otherwise pick from top moves overall
                    weighted_moves.sort(key=lambda x: x[1], reverse=True)
                    # Take best move 60% of time, random from top 3 otherwise
                    if random.random() < 0.6 and weighted_moves:
                        move = weighted_moves[0][0]
                    else:
                        top_n = min(3, len(weighted_moves))
                        idx = random.randint(0, top_n-1)
                        move = weighted_moves[idx][0]
                
                # Apply selected move
                move_score, sim_field = fake_place(move[0], move[1], move[2].coord_array, sim_field)
                score += move_score
                
                # Give substantial extra bonus for line clears
                if move_score > len(move[2].coord_array):
                    lines_cleared = (move_score - len(move[2].coord_array)) / 10
                    line_clears += lines_cleared
                    score += lines_cleared * 20  # Doubled bonus for line clears
                
                # Remove used block
                block_to_remove = move[2]
                for i, block in enumerate(available_blocks):
                    if block is block_to_remove:
                        available_blocks.pop(i)
                        break
                
                # Generate new blocks if needed
                if not available_blocks:
                    from blocks import BLOCKS
                    blocks_def = BLOCKS()
                    block_indices = [random.randint(0, len(blocks_def.block_list) - 1) for _ in range(3)]
                    block_coords = [blocks_def.block_list[i] for i in block_indices]
                    available_blocks = [SimpleBlock(coords) for coords in block_coords]
                
                steps += 1
            
            # Final board evaluation with additional line-clearing emphasis
            final_heuristic = AI.get_heuristic_score(sim_field) 
            score += final_heuristic
            
            # Reward for game longevity
            score += steps * 2
            
            # Massive bonus for total line clears achieved
            score += line_clears * 25  # Increased from 15
            
            return score
            
        def _evaluate_potential_lines(self, field):
            """Score the board based on potential for future line clears with more emphasis on nearly complete lines"""
            bonus = 0
            
            # Check rows - graduated values for rows with many filled cells
            for row in range(10):
                filled = sum(1 for cell in field[row] if cell == 1)
                # More aggressive bonus curve starting from 6+ filled cells
                if filled >= 6:
                    bonus += (filled - 5) ** 2  # Exponential bonus
                
            # Check columns
            for col in range(10):
                filled = sum(1 for row in range(10) if field[row][col] == 1)
                if filled >= 6:
                    bonus += (filled - 5) ** 2
                    
            # Look for patterns of cells that could form lines
            row_streak_bonus = self._evaluate_streaks(field)
            bonus += row_streak_bonus
            
            return bonus * 2.5  # Increased weight
        
        def _evaluate_streaks(self, field):
            """Evaluate continuous streaks of filled cells which could lead to line clears"""
            bonus = 0
            
            # Rows
            for y in range(10):
                streak_length = 0
                for x in range(10):
                    if field[y][x] == 1:
                        streak_length += 1
                    else:
                        # Only count longer streaks
                        if streak_length >= 3:
                            bonus += streak_length * 0.5
                        streak_length = 0
                if streak_length >= 3:
                    bonus += streak_length * 0.5
                    
            # Columns
            for x in range(10):
                streak_length = 0
                for y in range(10):
                    if field[y][x] == 1:
                        streak_length += 1
                    else:
                        if streak_length >= 3:
                            bonus += streak_length * 0.5
                        streak_length = 0
                if streak_length >= 3:
                    bonus += streak_length * 0.5
                    
            return bonus

        def _evaluate_connected_spaces(self, field):
            """Calculate bonus for having larger connected empty regions"""
            visited = set()
            total_bonus = 0
            
            for y in range(10):
                for x in range(10):
                    if field[y][x] == 0 and (x, y) not in visited:
                        # Do BFS to find connected region size
                        size = 0
                        queue = [(x, y)]
                        while queue:
                            cx, cy = queue.pop(0)
                            if (cx, cy) in visited:
                                continue
                            visited.add((cx, cy))
                            size += 1
                            
                            # Check neighbors
                            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                                nx, ny = cx + dx, cy + dy
                                if 0 <= nx < 10 and 0 <= ny < 10 and field[ny][nx] == 0 and (nx, ny) not in visited:
                                    queue.append((nx, ny))
                        
                        # Bonus formula: larger connected regions get disproportionately more points
                        if size > 1:
                            region_bonus = size * (1 + size/20)
                            total_bonus += region_bonus
                    
            return total_bonus / 10  # Scale down the bonus

        def _count_hard_corners(self, field):
            """Count isolated empty spaces and corner formations that are hard to fill"""
            count = 0
            for y in range(10):
                for x in range(10):
                    if field[y][x] == 0:
                        # Check surrounding cells
                        surroundings = []
                        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < 10 and 0 <= ny < 10:
                                surroundings.append(field[ny][nx])
                            else:
                                surroundings.append(1)  # Count borders as filled
                        
                        # Count filled neighbors
                        filled_neighbors = sum(surroundings)
                        
                        # Check for corner cases
                        if filled_neighbors >= 3:
                            count += 2  # Hard to fill corner
                        elif filled_neighbors == 2:
                            # Check if the filled cells are adjacent (making an L shape)
                            if surroundings[0] == 1 and surroundings[1] == 1:  # Top and Right
                                count += 1
                            elif surroundings[1] == 1 and surroundings[2] == 1:  # Right and Bottom
                                count += 1
                            elif surroundings[2] == 1 and surroundings[3] == 1:  # Bottom and Left
                                count += 1
                            elif surroundings[3] == 1 and surroundings[0] == 1:  # Left and Top
                                count += 1
            
            return count

        def _is_edge_placement(self, x, y, coordinates):
            """Check if a piece is placed along an edge or in a corner"""
            for coord in coordinates:
                tx = x + coord[0]
                ty = y + coord[1]
                if tx == 0 or tx == 9 or ty == 0 or ty == 9:
                    return True
            return False

        def _expand(self, node):
            """Expand a node by adding potential moves as children"""
            # If no valid moves or node is terminal
            if not hasattr(node, 'field') or node.field is None:
                return node
                
            # Get valid moves for current state
            if node.move_played:  # If this isn't the root
                # Clone current blocks minus the one we just placed
                blocks = [b for b in self.main.game.current_blocks if b is not node.move_played[2]]
                # If we've used all blocks, regenerate a new set
                if not blocks:
                    from blocks import BLOCKS
                    blocks_def = BLOCKS()
                    # In real game these would be newly generated blocks
                    block_indices = [random.randint(0, len(blocks_def.block_list) - 1) for _ in range(3)]
                    block_coords = [blocks_def.block_list[i] for i in block_indices]
                    blocks = [SimpleBlock(coords) for coords in block_coords]
            else:
                blocks = self.main.game.current_blocks
                
            # Find all valid moves with these blocks
            valid_moves = []
            for y in range(10):
                for x in range(10):
                    for block in blocks:
                        if fake_fits(x, y, block.coord_array, node.field):
                            valid_moves.append([x, y, block])
            
            # If no moves or all already explored
            if not valid_moves:
                return node
                
            # Choose a move randomly (we could be more strategic here)
            move = random.choice(valid_moves)
            
            # Create child node with the new field after applying the move
            score, new_field = fake_place(move[0], move[1], move[2].coord_array, 
                                        [row[:] for row in node.field])
            
            child = MCTSNode(new_field, self.main, node)
            child.move_played = move
            child.immediate_score = score
            child.cleared_lines = score > len(move[2].coord_array)  # Did this move clear any lines?
            
            node.children.append(child)
            return child
            
        def _backpropagate(self, node, reward):
            """Update statistics for all nodes in the path"""
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent


class MCTSNode:
    def __init__(self, field, _main, parent=None):
        self.field = field
        self.main = _main
        self.parent = parent
        self.children = []
        self.move_played = None
        self.immediate_score = 0  # Track immediate score from move
        self.visits = 0
        self.value = 0.0
