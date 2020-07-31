import random
from tkinter import *
from random import shuffle, randint
import copy
import math
import time
from itertools import permutations

move = []
moves = [None, None, None]
move_count = 0
rgb = False
last_block = None

def clean_field(field1, field2):
    field = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    for line in range(0, 10):
        for i in range(0, 10):
            if field1[line][i] == 0 or field2[line][i] == 0:
                field[line][i] = 0
            else:
                field[line][i] = 1
    return field

def fake_place(x, y, coordinates, field):
    for index in range(0, len(coordinates)):
        set_fake_filled(x + coordinates[index][0], y + coordinates[index][1], 1, field)
    points = len(coordinates)
    p, field1 = check_fake_lines(field)
    points += len(p) * 10
    p, field2 = check_fake_columns(field)
    points += len(p) * 10
    return points, clean_field(field1, field2)


class Main:

    def __init__(self):
        self.window = Tk()
        self.window.title("1010")
        self.window.geometry("600x750")
        self.window.configure(background='#474747')
        self.window.resizable(False, False)

        self.game = Game(self)
        self.AI = AI("greedy")
        self.last_x = None
        self.last_y = None
        self.last_preview = []

        self.points_label = Label(self.window, font=("Segoe UI Light", 24), bg="#474747", fg="lightgray")
        self.points_label["text"] = "0"
        self.points_label.place(x=(300 - self.points_label.winfo_width() / 2), y=10)

        self.canvas = Canvas(self.window, width=500, height=500, bg="lightgray", highlightthickness=0)
        self.canvas.bind("<Button-1>", self.canvas_click)
        # self.canvas.bind("<Motion>", self.render_preview)
        self.canvas.bind("<Leave>", self.remove_last_values)
        self.canvas.place(x=50, y=75)

        self.lose_img = PhotoImage(file='./resources/LoseScreenOverlay.gif')
        self.img = PhotoImage(file='./resources/DragAndDropOverlay.gif')
        self.bc_overlay = PhotoImage(file='./resources/BlockCanvasOverlay.gif')

        self.block_canvas = Canvas(self.window, width=500, height=125, bg="lightgray", highlightthickness=0)
        self.block_canvas.place(x=50, y=525 + 50 + 25)

        self.block_canvas.create_image(0, 0, image=self.bc_overlay, anchor="nw")
        self.img_id = self.canvas.create_image(0, 0, image=self.img, anchor="nw")

        self.game.generate_blocks()
        self.render_current_blocks()

        # GUILoseScreen(self.window, self.game, self.lose_img)
        # self.AI.play_once(main=self)
        self.window.mainloop()

    def canvas_click(self, event):
        # x = int(event.x / 50)
        # y = int(event.y / 50)
        self.AI.play_once(fake_main=self)
        global move
        x = move[0]
        y = move[1]
        self.game.selected_block = move[2]
        if (x < 10) and (y < 10):
            coordinates = self.game.selected_block.coord_array
            # if self.game.fits(x, y, coordinates):
            self.place(x, y, coordinates)
            block = self.game.selected_block
            block.destroy()
            self.game.selected_block = None
            self.game.current_blocks.remove(block)
            if len(self.game.current_blocks) == 0:
                self.game.generate_blocks()
                self.render_current_blocks()

            if len(self.game.check_lines()) > 0:
                for lines in self.game.check_lines():
                    self.game.clear_line(lines)
                    for i in range(0, 10):
                        self.clear_rect_on_coordinates(i, lines)

            if len(self.game.check_columns()) > 0:
                for columns in self.game.check_columns():
                    self.game.clear_column(columns)
                    for i in range(0, 10):
                        self.clear_rect_on_coordinates(columns, i)

            if not self.game.is_action_possible():
                GUILoseScreen(self.window, self.game, self.lose_img)

    # def render_preview(self, event):
    #     # x = int(event.x / 50)
    #     # y = int(event.y / 50)
    #
    #     global move
    #     x = move[0]
    #     y = move[1]
    #     self.game.selected_block = move[2]
    #     if self.last_x != x or self.last_y != y:
    #         self.last_x = x
    #         self.last_y = y
    #         if self.game.selected_block is not None and 0 <= x < 10 and 0 <= y < 10:
    #             if self.game.fits(x, y, self.game.selected_block.coord_array):
    #                 for index in range(0, len(self.last_preview)):
    #                     lx = self.last_preview[index][0]
    #                     ly = self.last_preview[index][1]
    #                     if self.game.field[ly][lx] == 0:
    #                         self.draw_rect(self.last_preview[index][0], self.last_preview[index][1], "lightgray")
    #                 if self.game.selected_block is not None:
    #                     ca = self.game.selected_block.coord_array
    #                     self.last_preview = []
    #                     for index in range(0, len(ca)):
    #                         tx = x + ca[index][0]
    #                         ty = y + ca[index][1]
    #                         if tx < 10 and ty < 10:
    #                             self.draw_rect(tx, ty, "yellow")
    #                             self.last_preview.append([x + ca[index][0], y + ca[index][1]])

    def place(self, x, y, coordinates):
        colors = ["lightblue", "pink", "red", "black", "yellow"]
        global rgb
        if not rgb:
            rand = 2
        else:
            rand = randint(0, len(colors) - 1)
        global last_block
        if last_block is not None:
            for index in range(0, len(last_block[2])):
                self.draw_rect_on_coordinates(last_block[0] + last_block[2][index][0], last_block[1] + last_block[2][index][1], "orange")
        last_block = [x, y, coordinates]

        for index in range(0, len(coordinates)):
            self.draw_rect_on_coordinates(x + coordinates[index][0], y + coordinates[index][1], colors[rand])
            self.game.set_filed(x + coordinates[index][0], y + coordinates[index][1], 1)

    def remove_last_values(self, event):
        self.last_x = None
        self.last_y = None
        for index in range(0, len(self.last_preview)):
            lx = self.last_preview[index][0]
            ly = self.last_preview[index][1]
            if self.game.field[ly][lx] == 0:
                self.draw_rect(self.last_preview[index][0], self.last_preview[index][1], "lightgray")

    def draw_rect_on_coordinates(self, x, y, color):
        self.draw_rect(x, y, color)

    def clear_rect_on_coordinates(self, x, y):
        self.draw_rect(x, y, "lightgray")

    def draw_rect(self, x, y, color):
        x = x * 50
        y = y * 50
        self.canvas.create_rectangle(x, y, x + 50, y + 50, fill=color, outline="")
        self.restore_grid(self.img_id)

    def render_current_blocks(self):
        for index in range(0, len(self.game.current_blocks)):
            c = self.game.current_blocks[index].get_block_canvas()
            c.place(x=50 + 166 * (index + 1) - 83 - int(c["width"]) / 2, y=75 + 500 + 25 + (62 - int(c["height"]) / 2))

    def restore_grid(self, img_id):
        self.img_id = self.canvas.create_image(0, 0, image=self.img, anchor="nw")
        self.canvas.delete(img_id)


class GUILoseScreen:
    def __init__(self, window, game, lose_img):
        canvas = Canvas(window, width=600, height=725, bg="#474747", highlightthickness=0)
        canvas.create_image(0, 0, image=lose_img, anchor="nw")
        canvas.place(x=0, y=0)


def check_fake_lines(field):
    lines = []
    for line in range(0, 10):
        flag = 1
        for i in range(0, 10):
            if field[line][i] != 1:
                flag = 0
                break
        if flag == 1:
            lines.append(line)
            for i in range(0, 10):
                field[line][i] = 0
    return lines, field


def check_fake_columns(field):
    columns = []
    for column in range(0, 10):
        flag = 1
        for i in range(0, 10):
            if field[i][column] != 1:
                flag = 0
                break
        if flag == 1:
            columns.append(column)
            for i in range(0, 10):
                field[i][column] = 0
    return columns, field


def set_fake_filled(x, y, full, fake_field):
    fake_field[y][x] = full
    return fake_field


def fake_fits(x, y, coordinates, field):
    for index in range(0, len(coordinates)):
        tx = x + coordinates[index][0]
        ty = y + coordinates[index][1]

        if 0 <= tx < 10 and 0 <= ty < 10:
            if field[ty][tx] == 1:
                return False
        else:
            return False
    return True


def valid_moves_for_block_on_field(block, field):
    moves = []
    for y in range(0, len(field)):
        for x in range(0, len(field[y])):
            if fake_fits(x, y, block.coord_array, field):
                moves.append([x, y, block])
    return moves


class Game:
    def __init__(self, gui):
        self.gui = gui
        self.field = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        self.points = 0
        self.blocks = BLOCKS()
        self.current_blocks = []
        self.selected_block = None

    def check_lines(self):
        lines = []
        for line in range(0, 10):
            flag = 1
            for i in range(0, 10):
                if self.field[line][i] != 1:
                    flag = 0
                    break
            if flag == 1:
                lines.append(line)
        return lines

    def check_columns(self):
        columns = []
        for column in range(0, 10):
            flag = 1
            for i in range(0, 10):
                if self.field[i][column] != 1:
                    flag = 0
                    break
            if flag == 1:
                columns.append(column)
        return columns

    def get_points(self):
        return self.points

    def add_points(self, points):
        self.points += points
        self.gui.points_label["text"] = str(self.points)
        self.gui.points_label.place(x=(300 - self.gui.points_label.winfo_width() / 2), y=10)

    def clear_line(self, index):
        for i in range(0, 10):
            self.set_filed(i, index, 0)

    def clear_column(self, index):
        for i in range(0, 10):
            self.set_filed(index, i, 0)

    def set_filed(self, x, y, full):
        self.add_points(1)
        self.field[y][x] = full

    def generate_blocks(self):
        self.current_blocks = []
        for i in range(0, 3):
            self.current_blocks.append(Block(randint(0, len(self.blocks.block_list) - 1), self.blocks, self.gui))

    def fits(self, x, y, coordinates):
        for index in range(0, len(coordinates)):
            tx = x + coordinates[index][0]
            ty = y + coordinates[index][1]

            if 0 <= tx < 10 and 0 <= ty < 10:
                if self.field[ty][tx] == 1:
                    return False
            else:
                return False
        return True

    def is_action_possible(self):
        for y in range(0, len(self.field)):
            for x in range(0, len(self.field[y])):
                for block in self.current_blocks:
                    if self.fits(x, y, block.coord_array):
                        return True
        return False

    def valid_moves(self):
        moves = []
        for y in range(0, len(self.field)):
            for x in range(0, len(self.field[y])):
                for block in self.current_blocks:
                    if self.fits(x, y, block.coord_array):
                        moves.append([x, y, block])
        return moves

    @staticmethod
    def valid_moves_on_field(field, blocks):
        moves = []
        for y in range(0, len(field)):
            for x in range(0, len(field[y])):
                for block in blocks:
                    if fake_fits(x, y, block.coord_array,field):
                        moves.append([x, y, block])
        return moves

    def valid_moves_for_block(self, block):
        moves = []
        for y in range(0, len(self.field)):
            for x in range(0, len(self.field[y])):
                # print(block)
                if self.fits(x, y, block.coord_array):
                    moves.append([x, y, block])
        return moves

    def copy_field(self):
        return copy.deepcopy(self.field)


class Block:
    def __init__(self, block_list_index, blocks, gui):
        self.block_list_index = block_list_index
        self.coord_array = blocks.block_list[block_list_index]
        self.gui = gui
        self.window = gui.window
        self.height = 0
        self.width = 0
        self.width_neg = 0
        self.set_measurement()
        self.canvas = self.__create_block_canvas()

    def set_measurement(self):
        width_pos = 0
        width_neg = 0
        height = 0
        for index in range(0, len(self.coord_array)):
            x1 = self.coord_array[index][0] * 25
            y1 = self.coord_array[index][1] * 25

            if x1 >= 0:
                if x1 + 25 > width_pos:
                    width_pos = x1 + 25
            elif x1 * -1 > width_neg:
                width_neg = (x1 * -1)

            if y1 + 25 > height:
                height = y1 + 25
        self.height = height
        self.width = width_pos + width_neg
        self.width_neg = width_neg

    def get_block_canvas(self):
        return self.canvas

    def __create_block_canvas(self):
        canvas = Canvas(self.window, width=self.width, height=self.height, bg="lightgray", highlightthickness=0)
        canvas.bind("<Button-1>", self.select_block)
        for index in range(0, len(self.coord_array)):
            x1 = self.coord_array[index][0] * 25
            y1 = self.coord_array[index][1] * 25
            canvas.create_rectangle(x1 + self.width_neg, y1, x1 + 25 + self.width_neg, y1 + 25, fill="orange",
                                    outline="")

        return canvas

    def select_block(self, event):
        selected_block = self.gui.game.selected_block
        if selected_block is not None and selected_block is not self:
            selected_block.remove_outline()
        self.gui.game.selected_block = self
        self.canvas["highlightthickness"] = 1

    def remove_outline(self):
        self.canvas["highlightthickness"] = 0

    def destroy(self):
        self.canvas.destroy()


class AI:
    AItype = ''

    def __init__(self, AItype):
        self.AItype = AItype

    def play_once(self, fake_main):
        global move
        if self.AItype == 'greedy':
            move = self.get_greedy_move(fake_main=fake_main)
        elif self.AItype == 'MCTS':
            ai = self.MCTSAI(_main=fake_main)
            move = ai.get_MCTS_move()
        if self.AItype == 'heuristic':
            move = self.get_heuristic_move(fake_main=fake_main)
            # main.place(move[0], move[1], move[2].coord_array)
            # if len(main.game.current_blocks) == 0:
            #     main.game.generate_blocks()
            #     main.render_current_blocks()
            #
            # if len(main.game.check_lines()) > 0:
            #     for lines in main.game.check_lines():
            #         main.game.clear_line(lines)
            #         for i in range(0, 10):
            #             main.clear_rect_on_coordinates(i, lines)
            #
            # if len(main.game.check_columns()) > 0:
            #     for columns in main.game.check_columns():
            #         main.game.clear_column(columns)
            #         for i in range(0, 10):
            #             main.clear_rect_on_coordinates(columns, i)
            # # time.sleep(2)

        # if not main.game.is_action_possible():
        #     GUILoseScreen(main.window, main.game, main.lose_img)

    @staticmethod
    def get_greedy_move(fake_main):
        moves = fake_main.game.valid_moves()
        shuffle(moves)
        best_move = 0
        best_score = 0
        for move_ in moves:
            field_copy = fake_main.game.copy_field()
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
            field_copy = fake_main.game.copy_field()
            if type(field_copy) == 'NoneType':
                print("aaaaaaaaaaaaaaaaaaaaaaaaaa")
            temp_score = 0
            move_list_of_perm = []
            for block in perm:
                temp_moves = valid_moves_for_block_on_field(block, field_copy)
                if len(temp_moves) == 0:
                    break
                block_score = -1
                block_move = None
                block_field = None
                shuffle(temp_moves)
                for move_ in temp_moves:
                    _, field = fake_place(x=move_[0], y=move_[1], coordinates=move_[2].coord_array, field=copy.deepcopy(field_copy))
                    heu_score = self.get_heuristic_score(field)
                    if heu_score+_ > block_score:
                        block_score = heu_score+_
                        block_move = move_
                        block_field = copy.deepcopy(field)
                move_list_of_perm.append(block_move)
                temp_score += block_score
                field_copy = copy.deepcopy(block_field)
            if temp_score > best_score:
                best_score = temp_score
                best_move_perm = move_list_of_perm
        global moves
        moves = best_move_perm

    @staticmethod
    def get_heuristic_score(field):
        score = 0
        for y in range(0, len(field)):
            for x in range(0, len(field[y])):
                if field[y][x] == 0:
                    score += 1
                if y != 0:
                    if field[y-1][x] == 0:
                        score += 2
                        if x != 0:
                            if field[y-1][x-1] == 0:
                                score += 4

                        if x != len(field[y])-1:
                            if field[y-1][x+1] == 0:
                                score += 4
                if y != len(field)-1:
                    if field[y+1][x] == 0:
                        score += 2
                        if x != 0:
                            if field[y+1][x-1] == 0:
                                score += 4

                        if x != len(field[y])-1:
                            if field[y+1][x+1] == 0:
                                score += 4
                if x != 0:
                    if field[y][x - 1] == 0:
                        score += 2

                if x != len(field[y])-1:
                    if field[y][x + 1] == 0:
                        score += 2
        return score

    def get_heuristic_move(self, fake_main):
        global moves
        if moves[0] is not None:
            move = moves[0]
            moves[0] = None
            return move
        if moves[1] is not None:
            move = moves[1]
            moves[1] = None
            return move
        if moves[2] is not None:
            move = moves[2]
            moves[2] = None
            return move
        self.get_heuristic_moves(fake_main)
        return self.get_heuristic_move(fake_main)

    class MCTSAI:
        level = 2
        main = None

        def __init__(self, _main):
            self.main = _main

        def select_best_node(self, root):
            node = root
            while len(node.children) != 0:
                node = self.find_best_node(node)
            return node

        def find_best_node(self, node):
            parent_visit = node.state.visit_count
            max_uct = 0
            max_uct_child = None
            for child in node.children:
                uct = self.uct(parent_visit, child.state.board_score, child.state.visit_count)
                if uct > max_uct:
                    max_uct = uct
                    max_uct_child = child
            return max_uct_child

        @staticmethod
        def uct(total_visit, node_win_score, node_visit):
            if node_visit == 0:
                return sys.maxsize
            return (node_win_score / node_visit) + 1.41 + math.sqrt(math.log(total_visit) / node_visit)

        def expand_node_origin(self, node, blocks, block):
            for move_ in valid_moves_for_block_on_field(blocks[block], node.state.field):
                _, field = fake_place(x=move_[0], y=move_[1], coordinates=move_[2].coord_array,
                                      field=copy.copy(node.state.field))
                new_node = AI.MCTSNode(_main=self.main, field=field, state=AI.MCTSState(_main=self.main,
                                                                                        field=field, state=node.state))
                new_node.parent = node
                node.children.append(new_node)
                new_node.move_played = move_
                if block < 2:
                    self.expand_node_origin(new_node, blocks, block + 1)

        def expand_node(self, node):
            for move_ in self.main.game.valid_moves():
                _, field = fake_place(x=move_[0], y=move_[1], coordinates=move_[2].coord_array,
                                      field=copy.copy(node.state.field))
                new_node = AI.MCTSNode(_main=self.main, field=field, state=AI.MCTSState(_main=self.main,
                                                                                        field=field, state=node.state))
                new_node.parent = node
                node.children.append(new_node)
                new_node.move_played = move_

        def back_propagation(self, node, score):
            temp = AI.MCTSNode(_main=self.main, field=node.state.field, node=node)
            while temp is not None:
                temp.state.add_visit()
                temp.state.add_score(score)
                temp = temp.parent

        def get_next_moves(self):
            roots = []
            perms = []
            perms.append(self.main.game.current_blocks)
            for i in range(2):
                perms.append(self.get_3_blocks())
            for i in range(3):
                roots.append(AI.MCTSNode(_main=self.main, field=self.main.game.copy_field(),
                                         state=AI.MCTSState(_main=self.main, field=self.main.game.copy_field())))
                self.expand_node_origin(node=roots[i], blocks=perms[i], block=0)
            start = time.time()
            j = 0
            while start + 2000 > time.time():
                for i in range(3):
                    node = self.select_best_node(roots[i])
                    if node is roots[i] and j == 1:
                        continue
                    elif node is roots[i]:
                        continue

                    exploration = AI.MCTSNode(_main=self.main, field=self.main.game.copy_field(), node=node)
                    if len(node.children) > 0:
                        exploration = random.choice(node.children)
                    score = self.sim_random_play(exploration)
                    self.back_propagation(exploration, score)

            winner = roots[0]
            evaluated_count = 0
            i = 0
            for root in roots:
                if root.state.board_score > winner.state.board_score:
                    winner = root
                    i += 1
                evaluated_count += root.state.visit_count
            print("Winner:" + str(i))
            for node in winner.children:
                print("Visit count" + str(node.state.visit_count) + " " + str(
                    node.move_played) + " with the score: " + str(node.state.board_score))
            print("Evaluation count: " + str(evaluated_count))
            t0 = winner.get_max_score_child()
            global moves
            moves[0] = t0.move_played
            if len(t0.children) != 0:
                t1 = t0.get_max_score_child()
                moves[1] = t1.move_played
                if len(t1.children) != 0:
                    t2 = t1.get_max_score_child()
                    moves[2] = t2.move_played
            print(moves)

        def sim_random_play(self, node):
            temp_node = main.AI.MCTSNode(field=node.state.field, _main=self.main, node=node)
            temp_state = temp_node.state
            if len(main.game.current_blocks):
                temp_state.board_score = -1
                return 0
            i = 0
            while temp_state.random_play():
                i += 1
            return i

        def get_3_blocks(self):
            blocks = []
            for i in range(3):
                blocks.append(Block(randint(0, len(self.main.game.blocks.block_list) - 1),
                                    self.main.game.blocks, self.main.game.gui))
            return blocks

        def get_MCTS_move(self):
            global moves
            if moves[0] is not None:
                move = moves[0]
                moves[0] = None
                return move
            if moves[1] is not None:
                move = moves[1]
                moves[1] = None
                return move
            if moves[2] is not None:
                move = moves[2]
                moves[2] = None
                return move
            self.get_next_moves()
            return self.get_MCTS_move()

    class MCTSNode:
        state = None
        children = []
        parent = None
        move_played = []
        field = []
        main = None

        def __init__(self, field, _main, state=None, parent=None, children=None, node=None):
            if node is not None:
                self.children = []
                self.state = AI.MCTSState(field=field.copy(), _main=_main, state=node.state)
                if node.get_parent() is not None:
                    self.parent = node.parent
                older_children = node.children()
                for child in older_children:
                    self.children.append(child)
            else:
                if state is not None:
                    self.state = state
                else:
                    self.state = AI.MCTSState(field=field.copy(), _main=_main)
                if children is not None:
                    self.children = children
                if parent is None:
                    self.parent = parent
            self.main = _main

        def get_max_score_child(self):
            max_child = None
            max_score = 0
            for child in self.children:
                score = child.state.board_score
                if score > max_score:
                    max_child = child
                    max_score = score
            return max_child

        def get_random_child(self):
            return self.children[randint(0, len(self.children) - 1)]

    class MCTSState:
        visit_count = 0
        board_score = 0
        field = []

        def __init__(self, field, _main, state=None):
            if state is not None:
                self.visit_count = state.visit_count
                self.board_score = state.board_score
                self.field = state.field
            else:
                self.field = field
            self.main = _main

        def add_visit(self):
            self.visit_count += 1

        def add_score(self, score):
            self.board_score += score

        def random_play(self):
            random_tile = Block(randint(0, len(self.main.game.blocks.block_list) - 1),
                                self.main.game.blocks, self.main.game.gui)
            _moves = main.game.valid_moves_for_block(random_tile)
            if len(_moves) == 0:
                return False
            _random = randint(0, len(_moves) - 1)
            _move = _moves[_random]

            _, _ = fake_place(x=_move[0], y=_move[1], coordinates=_move[2].coord_array,
                              field=copy.copy(self.field))
            return True


class BLOCKS:  # enum

    def __init__(self):
        self.block_list = [
            [[0, 0]],
            [[0, 0], [0, 1]],
            [[0, 0], [0, 1], [0, 2]],
            [[0, 0], [0, 1], [0, 2], [0, 3]],
            [[0, 0], [0, 1], [1, 0]],
            [[0, 0], [1, 0], [2, 0]],
            [[0, 0], [1, 0], [2, 0], [3, 0]],
            [[0, 0], [0, 1], [1, 1]],
            [[0, 0], [1, 0], [1, 1]],
            [[1, 0], [1, 1], [0, 1]],
            [[0, 0], [0, 1], [1, 0], [1, 1]],
            [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2]],
            [[0, 0], [1, 0]],
            [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]],
            [[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]],
        ]


main = Main()
