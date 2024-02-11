from copy import deepcopy
import time
import math
import cv2 as cv
import datetime
import os
import numpy as np
import pickle
import serial

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

ansi_black = "\u001b[30m"
ansi_red = "\u001b[31m"
ansi_green = "\u001b[32m"
ansi_yellow = "\u001b[33m"
ansi_blue = "\u001b[34m"
ansi_magenta = "\u001b[35m"
ansi_cyan = "\u001b[36m"
ansi_white = "\u001b[37m"
ansi_reset = "\u001b[0m"

VIDEO_SOURCE = 0


class Node:
    def __init__(self, board, move=None, parent=None, value=None):
        self.board = board
        self.value = value
        self.move = move
        self.parent = parent

    def get_children(self, minimizing_player, mandatory_jumping):
        current_state = deepcopy(self.board)
        available_moves = []
        children_states = []
        big_letter = ""
        queen_row = 0
        if minimizing_player is True:
            available_moves = Checkers.find_available_moves(current_state, mandatory_jumping)
            big_letter = "C"
            queen_row = 7
        else:
            available_moves = Checkers.find_player_available_moves(current_state, mandatory_jumping)
            big_letter = "B"
            queen_row = 0
        for i in range(len(available_moves)):
            old_i = available_moves[i][0]
            old_j = available_moves[i][1]
            new_i = available_moves[i][2]
            new_j = available_moves[i][3]
            state = deepcopy(current_state)
            Checkers.make_a_move(state, old_i, old_j, new_i, new_j, big_letter, queen_row)
            children_states.append(Node(state, [old_i, old_j, new_i, new_j]))
        return children_states

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def get_board(self):
        return self.board

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent


class PoseEstimator:
    def __init__(self, grid_x: int = 7, grid_y: int = 7,
                 calibration_path: str =
                 r'D:\IngMagistrale\Smart Robotics\MrCrabRobotArm\camera_calibration\calibration.pkl'):
        self.calibration_path = calibration_path
        with open(self.calibration_path, 'rb') as file:
            data = pickle.load(file)
            self.mtx, self.dist = data
        self.GRID_X = grid_x
        self.GRID_Y = grid_y
        self.GRID_SHAPE = (self.GRID_X, self.GRID_Y)
        self.N_CELLS = self.GRID_SHAPE[0] * self.GRID_SHAPE[1]

        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((self.N_CELLS, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.GRID_X, 0:self.GRID_Y].T.reshape(-1, 2)
        self.axis = np.float32([[self.GRID_X, 0, 0], [0, self.GRID_Y, 0], [0, 0, -self.GRID_X]]).reshape(-1, 3)

        self.corners = None
        self.rvecs, self.tvecs = None, None

        self.cells = np.float32(
            [[-0.5, -0.5, 0], [0.5, -0.5, 0], [1.5, -0.5, 0], [2.5, -0.5, 0], [3.5, -0.5, 0], [4.5, -0.5, 0],
             [5.5, -0.5, 0], [6.5, -0.5, 0],
             [-0.5, 0.5, 0], [0.5, 0.5, 0], [1.5, 0.5, 0], [2.5, 0.5, 0], [3.5, 0.5, 0], [4.5, 0.5, 0], [5.5, 0.5, 0],
             [6.5, 0.5, 0],
             [-0.5, 1.5, 0], [0.5, 1.5, 0], [1.5, 1.5, 0], [2.5, 1.5, 0], [3.5, 1.5, 0], [4.5, 1.5, 0], [5.5, 1.5, 0],
             [6.5, 1.5, 0],
             [-0.5, 2.5, 0], [0.5, 2.5, 0], [1.5, 2.5, 0], [2.5, 2.5, 0], [3.5, 2.5, 0], [4.5, 2.5, 0], [5.5, 2.5, 0],
             [6.5, 2.5, 0],
             [-0.5, 3.5, 0], [0.5, 3.5, 0], [1.5, 3.5, 0], [2.5, 3.5, 0], [3.5, 3.5, 0], [4.5, 3.5, 0], [5.5, 3.5, 0],
             [6.5, 3.5, 0],
             [-0.5, 4.5, 0], [0.5, 4.5, 0], [1.5, 4.5, 0], [2.5, 4.5, 0], [3.5, 4.5, 0], [4.5, 4.5, 0], [5.5, 4.5, 0],
             [6.5, 4.5, 0],
             [-0.5, 5.5, 0], [0.5, 5.5, 0], [1.5, 5.5, 0], [2.5, 5.5, 0], [3.5, 5.5, 0], [4.5, 5.5, 0], [5.5, 5.5, 0],
             [6.5, 5.5, 0],
             [-0.5, 6.5, 0], [0.5, 6.5, 0], [1.5, 6.5, 0], [2.5, 6.5, 0], [3.5, 6.5, 0], [4.5, 6.5, 0], [5.5, 6.5, 0],
             [6.5, 6.5, 0]])

        self.projected_axis = None
        self.projected_cells = None

        self.BLU_BGR = (255, 0, 0)
        self.GREEN_BGR = (0, 255, 0)
        self.RED_BGR = (0, 0, 255)
        self.THICKNESS = 3

        self.in_board_error = np.array([10., 10.])

    def estimate_parameters(self, gray):
        ret, self.corners = cv.findChessboardCorners(gray, self.GRID_SHAPE, None)
        if ret:
            corners2 = cv.cornerSubPix(gray, self.corners, (11, 11), (-1, -1), self.criteria)

            ret, rvecs, tvecs = cv.solvePnP(self.objp, corners2, self.mtx, self.dist)
            self.rvecs, self.tvecs = rvecs, tvecs

            self.projected_cells, jac = cv.projectPoints(self.cells.reshape(-1, 3), self.rvecs, self.tvecs, self.mtx,
                                                         self.dist)
            self.projected_axis, jac = cv.projectPoints(self.axis, self.rvecs, self.tvecs, self.mtx, self.dist)
        else:
            raise SystemError("[POSE_ESTIMATOR] ERROR: Impossible to detect chessboard corners!")

    def estimate_point(self, point):
        imgpts, jac = cv.projectPoints(point.reshape(-1, 3), self.rvecs, self.tvecs, self.mtx, self.dist)
        return imgpts

    def estimate_cells(self):
        imgpts, jac = cv.projectPoints(self.cells.reshape(-1, 3), self.rvecs, self.tvecs, self.mtx, self.dist)
        return imgpts

    def find_nearest_cell(self, pixel):
        tmp = np.empty(shape=[1, 1, pixel.shape[0]])
        tmp[:, :] = pixel
        distances = np.linalg.norm(self.projected_cells - pixel, axis=2)
        nearest_idx = np.argmin(distances)
        nearest_cell = self.projected_cells[nearest_idx]

        i_pixel, j_pixel = -1, -1
        for i in range(self.GRID_X + 1):
            for j in range(self.GRID_Y):
                if i * (self.GRID_X+1) + j == nearest_idx:
                    i_pixel, j_pixel = i, j
                    break
            if i_pixel != -1 and j_pixel != -1:
                break
        if i_pixel == -1 or j_pixel == -1:
            raise ValueError("[POSE_ESTIMATOR] ERROR: something has gone wrong with the cell estimation!")

        return i_pixel, j_pixel

    def draw_axis(self, img):
        corner = tuple(self.corners[0].astype(int).ravel())
        img = cv.line(img, pt1=corner, pt2=tuple(self.projected_axis[0].astype(int).ravel()), color=self.BLU_BGR, thickness=self.THICKNESS)
        img = cv.line(img, pt1=corner, pt2=tuple(self.projected_axis[1].astype(int).ravel()), color=self.GREEN_BGR, thickness=self.THICKNESS)
        # img = cv.line(img, pt1=corner, pt2=tuple(self.projected_axis[2].astype(int).ravel()), color=self.RED_BGR, thickness=self.THICKNESS)
        return img

    def draw_points(self, img, points):
        for idx in range(points.shape[0]):
            img = cv.circle(img, tuple(points[idx].astype(int).ravel()), 5, (0, 255, 0), -1)

    @staticmethod
    def draw_point(img, point):
        return cv.circle(img, tuple(point.astype(int).ravel()), 5, (0, 255, 0), -1)

    def draw_cells(self, img):
        for idx in range(self.projected_cells.shape[0]):
            img = cv.circle(img, tuple(self.projected_cells[idx].astype(int).ravel()), 5, self.RED_BGR, -1)
        return img

    def get_projected_cells(self):
        return self.projected_cells

    def get_cells(self):
        return self.cells


class RoboticArm:
    def __init__(self, link1: float, link2: float, cells, port: str = 'COM5'):
        self.port = port
        self.ser = serial.Serial(port)
        print(f"[ROBOT] Connected to Robot on port {port}")

        self.link1 = link1  # length of first link (cm)
        self.link2 = link2  # length of second link (cm)

        self.cells = cells  # array shape (n_cells, 3) -> 3d position of each cell
        self.GRID_SIZE = 8

        self.OKAY_MSG = 'okay'

        self.SHOULDER_IDLE_ANGLE = 0
        self.ELBOW_IDLE_ANGLE = 0
        self.BASE_IDLE_ANGLE = 0

        self.REST_CODE = 0
        self.SHOULDER_CODE = 1
        self.ELBOW_CODE = 2
        self.ELBOW_UP_CODE = 3
        self.CLOSE_GRIPPER_CODE = 4
        self.OPEN_GRIPPER_CODE = 5
        self.BASE_CODE = 6

    def wait_until_msg(self, msg: str):
        while True:
            rec = self.ser.readline().decode().strip()
            if msg in rec:
                break
            else:
                print("Received message:", rec)

    def command(self, code: int, arg: int = 0, wait: bool = True):
        # Convert code and arg to bytes
        command_bytes = bytes([code])
        arg_bytes = bytes([arg])

        # Send the command code and argument to Arduino
        # TODO: da verificare che questa riga sia corretta
        self.ser.write(command_bytes + arg_bytes)

        print(f"[SERIAL] Command {code} with argument {arg} sent!")

        if wait:
            self.wait_until_msg(self.OKAY_MSG)

    def convert_cell_to_3d(self, cell):
        i, j = cell[0], cell[1]
        return self.cells[i * self.GRID_SIZE + j]

    def inverse_kinematics(self, x, y, z):
        s_angle, e_angle = 0, 0
        # TODO: implementare le formule

        return round(s_angle), round(e_angle)

    @staticmethod
    def compute_base_angle(x: float, y: float):
        angle = math.atan(y / x) * 180 / math.pi

        # TODO: non sono sicuro del significato e dell'utilità di questo blocco di codice
        if angle < 0:
            angle += 180
        if x < 0:  # adjusting angle when square is to the left of base.
            angle -= x*3

        return round(angle)

    def go_to_rest(self):
        self.command(self.ELBOW_CODE, self.ELBOW_IDLE_ANGLE)
        self.command(self.CLOSE_GRIPPER_CODE)
        self.command(self.SHOULDER_CODE, self.SHOULDER_IDLE_ANGLE)
        self.command(self.BASE_CODE, self.BASE_IDLE_ANGLE)

    def go_to(self, x, y, z):
        # calcolare angolo base
        base_angle = self.compute_base_angle(x, y)

        # allineare base
        self.command(self.BASE_CODE, base_angle)

        # cinematica inversa del 2r planar robot
        s_angle, e_angle = self.inverse_kinematics(x, y, z)

        # muovere spalla
        self.command(self.SHOULDER_CODE, s_angle)

        # muovere elbow
        self.command(self.ELBOW_CODE, e_angle)

    def move_from_to(self, start_cell, end_cell):
        # convert start_cell coordinates
        x_start, y_start, z_start = self.convert_cell_to_3d(start_cell)

        # convert end_cell coordinates
        x_end, y_end, z_end = self.convert_cell_to_3d(end_cell)

        # send command to Arduino to move to start cell
        self.go_to(x_start, y_start, z_start)

        # grab the piece
        self.command(self.CLOSE_GRIPPER_CODE)

        # lift elbow from the chessboard by a certain degree
        # TODO: forse sarebbe più comodo incorporare questo step nel comando del gripper
        self.command(self.ELBOW_UP_CODE)

        # send command to Arduino to move to end cell
        self.go_to(x_end, y_end, z_end)

        # release the piece
        self.command(self.OPEN_GRIPPER_CODE)

        self.command(self.ELBOW_UP_CODE)

        # send command to Arduino to move to idle state
        self.go_to_rest()

    def capture_piece(self, piece_cell):
        pass

    def promote_to_king(self, piece_cell):
        pass

    def move_king(self, start_cell, end_cell):
        pass

    def move(self, move_dict):
        if not isinstance(move_dict, dict):
            raise TypeError("[ROBOT] ERROR: The move can be performed only if a dictionary is passed as input!")

        if 'from' not in move_dict or 'to' not in move_dict:
            raise ValueError(f"[ROBOT] ERROR: Missing 'from' and 'to' key in move_dict: {move_dict} !!!")

        # muovi il pezzo del computer da from a to
        # TODO
        self.move_from_to(move_dict['from'], move_dict['to'])

        if 'capture' in move_dict:
            # rimuovi il pezzo catturato
            # TODO
            self.capture_piece(move_dict['capture'])

        if 'promotion' in move_dict:
            # chiedere all'utente di posizionare il pezzo di promozione
            # TODO
            while True:
                key = input(f"[ROBOT] Piece {move_dict['to']} got promoted. Please position the king-piece, "
                            f"then input [m] when done, in the console.")
                if key == 'm':
                    break
                else:
                    print("[ROBOT] Invalid key!")

            # TODO
            # posizionare il pezzo di promozione in to sopra al pezzo già presente
            self.promote_to_king(move_dict['to'])

        if 'king' in move_dict:
            # TODO: muovere king
            self.move_king(move_dict['from'], move_dict['to'])

        # only for DEBUG!!!
        val = input("[ROBOT] Waiting for robot actuation... press [m] when done.")
        print(val)


class Checkers:

    def __init__(self):
        self.matrix = [[], [], [], [], [], [], [], []]
        self.player_turn = True
        self.computer_pieces = 12
        self.player_pieces = 12
        self.available_moves = []
        self.mandatory_jumping = False

        for row in self.matrix:
            for i in range(8):
                row.append("---")
        self.position_computer()
        self.position_player()

        self.game_directory = ''
        self.counter_picture = 0

        self.cap = cv.VideoCapture(VIDEO_SOURCE)
        if not self.cap.isOpened():
            raise SystemError("[MAIN] ERROR: Impossible to open camera!")

        self.pose_estimator = PoseEstimator()
        self.calibration_frame = None
        self.model = YOLO(r'D:\IngMagistrale\Smart Robotics\MrCrabRobotArm\checkers_dataset\runs\detect\train2'
                          r'\weights\best.pt')

        self.robot = None
        self.old_matrix = self.matrix

    def position_computer(self):
        for i in range(3):
            for j in range(8):
                if (i + j) % 2 == 1:
                    self.matrix[i][j] = ("c" + str(i) + str(j))

    def position_player(self):
        for i in range(5, 8, 1):
            for j in range(8):
                if (i + j) % 2 == 1:
                    self.matrix[i][j] = ("b" + str(i) + str(j))

    def print_matrix(self):
        i = 0
        print()
        for row in self.matrix:
            print(i, end="  |")
            i += 1
            for elem in row:
                print(elem, end=" ")
            print()
        print()
        for j in range(8):
            if j == 0:
                j = "     0"
            print(j, end="   ")
        print("\n")

    def get_player_input(self):
        available_moves = Checkers.find_player_available_moves(self.matrix, self.mandatory_jumping)
        if len(available_moves) == 0:
            if self.computer_pieces > self.player_pieces:
                print(
                    ansi_red + "You have no moves left, and you have fewer pieces than the computer.YOU LOSE!" + ansi_reset)
                exit()
            else:
                print(ansi_yellow + "You have no available moves.\nGAME ENDED!" + ansi_reset)
                exit()
        self.player_pieces = 0
        self.computer_pieces = 0
        while True:

            coord1 = input("Which piece[i,j]: ")
            if coord1 == "":
                print(ansi_cyan + "Game ended!" + ansi_reset)
                exit()
            elif coord1 == "s":
                print(ansi_cyan + "You surrendered.\nCoward." + ansi_reset)
                exit()
            coord2 = input("Where to[i,j]:")
            if coord2 == "":
                print(ansi_cyan + "Game ended!" + ansi_reset)
                exit()
            elif coord2 == "s":
                print(ansi_cyan + "You surrendered.\nCoward." + ansi_reset)
                exit()
            old = coord1.split(",")
            new = coord2.split(",")

            if len(old) != 2 or len(new) != 2:
                print(ansi_red + "Illegal input" + ansi_reset)
            else:
                old_i = old[0]
                old_j = old[1]
                new_i = new[0]
                new_j = new[1]
                if not old_i.isdigit() or not old_j.isdigit() or not new_i.isdigit() or not new_j.isdigit():
                    print(ansi_red + "Illegal input" + ansi_reset)
                else:
                    move = [int(old_i), int(old_j), int(new_i), int(new_j)]
                    if move not in available_moves:
                        print(ansi_red + "Illegal move!" + ansi_reset)
                    else:
                        Checkers.make_a_move(self.matrix, int(old_i), int(old_j), int(new_i), int(new_j), "B", 0)
                        for m in range(8):
                            for n in range(8):
                                if self.matrix[m][n][0] == "c" or self.matrix[m][n][0] == "C":
                                    self.computer_pieces += 1
                                elif self.matrix[m][n][0] == "b" or self.matrix[m][n][0] == "B":
                                    self.player_pieces += 1
                        break

    @staticmethod
    def find_available_moves(board, mandatory_jumping):
        available_moves = []
        available_jumps = []
        for m in range(8):
            for n in range(8):
                if board[m][n][0] == "c":
                    if Checkers.check_moves(board, m, n, m + 1, n + 1):
                        available_moves.append([m, n, m + 1, n + 1])
                    if Checkers.check_moves(board, m, n, m + 1, n - 1):
                        available_moves.append([m, n, m + 1, n - 1])
                    if Checkers.check_jumps(board, m, n, m + 1, n - 1, m + 2, n - 2):
                        available_jumps.append([m, n, m + 2, n - 2])
                    if Checkers.check_jumps(board, m, n, m + 1, n + 1, m + 2, n + 2):
                        available_jumps.append([m, n, m + 2, n + 2])
                elif board[m][n][0] == "C":
                    if Checkers.check_moves(board, m, n, m + 1, n + 1):
                        available_moves.append([m, n, m + 1, n + 1])
                    if Checkers.check_moves(board, m, n, m + 1, n - 1):
                        available_moves.append([m, n, m + 1, n - 1])
                    if Checkers.check_moves(board, m, n, m - 1, n - 1):
                        available_moves.append([m, n, m - 1, n - 1])
                    if Checkers.check_moves(board, m, n, m - 1, n + 1):
                        available_moves.append([m, n, m - 1, n + 1])
                    if Checkers.check_jumps(board, m, n, m + 1, n - 1, m + 2, n - 2):
                        available_jumps.append([m, n, m + 2, n - 2])
                    if Checkers.check_jumps(board, m, n, m - 1, n - 1, m - 2, n - 2):
                        available_jumps.append([m, n, m - 2, n - 2])
                    if Checkers.check_jumps(board, m, n, m - 1, n + 1, m - 2, n + 2):
                        available_jumps.append([m, n, m - 2, n + 2])
                    if Checkers.check_jumps(board, m, n, m + 1, n + 1, m + 2, n + 2):
                        available_jumps.append([m, n, m + 2, n + 2])
        if mandatory_jumping is False:
            available_jumps.extend(available_moves)
            return available_jumps
        elif mandatory_jumping is True:
            if len(available_jumps) == 0:
                return available_moves
            else:
                return available_jumps

    @staticmethod
    def check_jumps(board, old_i, old_j, via_i, via_j, new_i, new_j):
        if new_i > 7 or new_i < 0:
            return False
        if new_j > 7 or new_j < 0:
            return False
        if board[via_i][via_j] == "---":
            return False
        if board[via_i][via_j][0] == "C" or board[via_i][via_j][0] == "c":
            return False
        if board[new_i][new_j] != "---":
            return False
        if board[old_i][old_j] == "---":
            return False
        if board[old_i][old_j][0] == "b" or board[old_i][old_j][0] == "B":
            return False
        return True

    @staticmethod
    def check_moves(board, old_i, old_j, new_i, new_j):

        if new_i > 7 or new_i < 0:
            return False
        if new_j > 7 or new_j < 0:
            return False
        if board[old_i][old_j] == "---":
            return False
        if board[new_i][new_j] != "---":
            return False
        if board[old_i][old_j][0] == "b" or board[old_i][old_j][0] == "B":
            return False
        if board[new_i][new_j] == "---":
            return True

    @staticmethod
    def calculate_heuristics(board):
        result = 0
        mine = 0
        opp = 0
        for i in range(8):
            for j in range(8):
                if board[i][j][0] == "c" or board[i][j][0] == "C":
                    mine += 1

                    if board[i][j][0] == "c":
                        result += 5
                    if board[i][j][0] == "C":
                        result += 10
                    if i == 0 or j == 0 or i == 7 or j == 7:
                        result += 7
                    if i + 1 > 7 or j - 1 < 0 or i - 1 < 0 or j + 1 > 7:
                        continue
                    if (board[i + 1][j - 1][0] == "b" or board[i + 1][j - 1][0] == "B") and board[i - 1][
                        j + 1] == "---":
                        result -= 3
                    if (board[i + 1][j + 1][0] == "b" or board[i + 1][j + 1] == "B") and board[i - 1][j - 1] == "---":
                        result -= 3
                    if board[i - 1][j - 1][0] == "B" and board[i + 1][j + 1] == "---":
                        result -= 3

                    if board[i - 1][j + 1][0] == "B" and board[i + 1][j - 1] == "---":
                        result -= 3
                    if i + 2 > 7 or i - 2 < 0:
                        continue
                    if (board[i + 1][j - 1][0] == "B" or board[i + 1][j - 1][0] == "b") and board[i + 2][
                        j - 2] == "---":
                        result += 6
                    if i + 2 > 7 or j + 2 > 7:
                        continue
                    if (board[i + 1][j + 1][0] == "B" or board[i + 1][j + 1][0] == "b") and board[i + 2][
                        j + 2] == "---":
                        result += 6

                elif board[i][j][0] == "b" or board[i][j][0] == "B":
                    opp += 1

        return result + (mine - opp) * 1000

    @staticmethod
    def find_player_available_moves(board, mandatory_jumping):
        available_moves = []
        available_jumps = []
        for m in range(8):
            for n in range(8):
                if board[m][n][0] == "b":
                    if Checkers.check_player_moves(board, m, n, m - 1, n - 1):
                        available_moves.append([m, n, m - 1, n - 1])
                    if Checkers.check_player_moves(board, m, n, m - 1, n + 1):
                        available_moves.append([m, n, m - 1, n + 1])
                    if Checkers.check_player_jumps(board, m, n, m - 1, n - 1, m - 2, n - 2):
                        available_jumps.append([m, n, m - 2, n - 2])
                    if Checkers.check_player_jumps(board, m, n, m - 1, n + 1, m - 2, n + 2):
                        available_jumps.append([m, n, m - 2, n + 2])
                elif board[m][n][0] == "B":
                    if Checkers.check_player_moves(board, m, n, m - 1, n - 1):
                        available_moves.append([m, n, m - 1, n - 1])
                    if Checkers.check_player_moves(board, m, n, m - 1, n + 1):
                        available_moves.append([m, n, m - 1, n + 1])
                    if Checkers.check_player_jumps(board, m, n, m - 1, n - 1, m - 2, n - 2):
                        available_jumps.append([m, n, m - 2, n - 2])
                    if Checkers.check_player_jumps(board, m, n, m - 1, n + 1, m - 2, n + 2):
                        available_jumps.append([m, n, m - 2, n + 2])
                    if Checkers.check_player_moves(board, m, n, m + 1, n - 1):
                        available_moves.append([m, n, m + 1, n - 1])
                    if Checkers.check_player_jumps(board, m, n, m + 1, n - 1, m + 2, n - 2):
                        available_jumps.append([m, n, m + 2, n - 2])
                    if Checkers.check_player_moves(board, m, n, m + 1, n + 1):
                        available_moves.append([m, n, m + 1, n + 1])
                    if Checkers.check_player_jumps(board, m, n, m + 1, n + 1, m + 2, n + 2):
                        available_jumps.append([m, n, m + 2, n + 2])
        if mandatory_jumping is False:
            available_jumps.extend(available_moves)
            return available_jumps
        elif mandatory_jumping is True:
            if len(available_jumps) == 0:
                return available_moves
            else:
                return available_jumps

    @staticmethod
    def check_player_moves(board, old_i, old_j, new_i, new_j):
        if new_i > 7 or new_i < 0:
            return False
        if new_j > 7 or new_j < 0:
            return False
        if board[old_i][old_j] == "---":
            return False
        if board[new_i][new_j] != "---":
            return False
        if board[old_i][old_j][0] == "c" or board[old_i][old_j][0] == "C":
            return False
        if board[new_i][new_j] == "---":
            return True

    @staticmethod
    def check_player_jumps(board, old_i, old_j, via_i, via_j, new_i, new_j):
        if new_i > 7 or new_i < 0:
            return False
        if new_j > 7 or new_j < 0:
            return False
        if board[via_i][via_j] == "---":
            return False
        if board[via_i][via_j][0] == "B" or board[via_i][via_j][0] == "b":
            return False
        if board[new_i][new_j] != "---":
            return False
        if board[old_i][old_j] == "---":
            return False
        if board[old_i][old_j][0] == "c" or board[old_i][old_j][0] == "C":
            return False
        return True

    def evaluate_states(self):
        t1 = time.time()
        current_state = Node(deepcopy(self.matrix))

        first_computer_moves = current_state.get_children(True, self.mandatory_jumping)
        if len(first_computer_moves) == 0:
            if self.player_pieces > self.computer_pieces:
                print(
                    ansi_yellow + "Computer has no available moves left, and you have more pieces left.\nYOU WIN!" + ansi_reset)
                exit()
            else:
                print(ansi_yellow + "Computer has no available moves left.\nGAME ENDED!" + ansi_reset)
                exit()
        dict = {}
        for i in range(len(first_computer_moves)):
            child = first_computer_moves[i]
            value = Checkers.minimax(child.get_board(), 4, -math.inf, math.inf, False, self.mandatory_jumping)
            dict[value] = child
        if len(dict.keys()) == 0:
            print(ansi_green + "Computer has cornered itself.\nYOU WIN!" + ansi_reset)
            exit()
        new_board = dict[max(dict)].get_board()
        move = dict[max(dict)].move
        self.matrix = new_board
        t2 = time.time()
        diff = t2 - t1
        print("Computer has moved (" + str(move[0]) + "," + str(move[1]) + ") to (" + str(move[2]) + "," + str(
            move[3]) + ").")
        print("It took him " + str(diff) + " seconds.")

        return move

    @staticmethod
    def minimax(board, depth, alpha, beta, maximizing_player, mandatory_jumping):
        if depth == 0:
            return Checkers.calculate_heuristics(board)
        current_state = Node(deepcopy(board))
        if maximizing_player is True:
            max_eval = -math.inf
            for child in current_state.get_children(True, mandatory_jumping):
                ev = Checkers.minimax(child.get_board(), depth - 1, alpha, beta, False, mandatory_jumping)
                max_eval = max(max_eval, ev)
                alpha = max(alpha, ev)
                if beta <= alpha:
                    break
            current_state.set_value(max_eval)
            return max_eval
        else:
            min_eval = math.inf
            for child in current_state.get_children(False, mandatory_jumping):
                ev = Checkers.minimax(child.get_board(), depth - 1, alpha, beta, True, mandatory_jumping)
                min_eval = min(min_eval, ev)
                beta = min(beta, ev)
                if beta <= alpha:
                    break
            current_state.set_value(min_eval)
            return min_eval

    @staticmethod
    def make_a_move(board, old_i, old_j, new_i, new_j, big_letter, queen_row):
        letter = board[old_i][old_j][0]
        i_difference = old_i - new_i
        j_difference = old_j - new_j
        if i_difference == -2 and j_difference == 2:
            board[old_i + 1][old_j - 1] = "---"

        elif i_difference == 2 and j_difference == 2:
            board[old_i - 1][old_j - 1] = "---"

        elif i_difference == 2 and j_difference == -2:
            board[old_i - 1][old_j + 1] = "---"

        elif i_difference == -2 and j_difference == -2:
            board[old_i + 1][old_j + 1] = "---"

        if new_i == queen_row:
            letter = big_letter
        board[old_i][old_j] = "---"
        board[new_i][new_j] = letter + str(new_i) + str(new_j)

    def has_player_move(self):
        available_moves = Checkers.find_player_available_moves(self.matrix, self.mandatory_jumping)
        has_move = True
        if len(available_moves) == 0:
            has_move = False
            if self.computer_pieces > self.player_pieces:
                print(
                    ansi_red + "You have no moves left, and you have fewer pieces than the computer.YOU LOSE!" + ansi_reset)
            else:
                print(ansi_yellow + "You have no available moves.\nGAME ENDED!" + ansi_reset)
        return has_move

    def wait_for_player_move(self):
        print("Press [m] to confirm the move, or [s] to quit in the visualization window...")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[PLAYER] Can't receive frame (stream end?). Exiting ...")
                break

            detection_frame, loc_frame = self.detection_board(np.copy(frame))

            cv.imshow("Game", frame)
            cv.imshow("Detection", detection_frame)
            self.pose_estimator.draw_axis(loc_frame)
            cv.imshow("Localization", loc_frame)

            key = cv.waitKey(5)
            if key == ord('m'):
                print("[PLAYER] Move confirmed.")
                cv.imwrite(self.game_directory + f'game_{self.counter_picture}.png', frame)
                cv.imwrite(self.game_directory + f'detection_{self.counter_picture}.png', detection_frame)
                cv.imwrite(self.game_directory + f'localization_{self.counter_picture}.png', loc_frame)
                return True
            elif key == ord('s'):
                print(ansi_cyan + "Coward." + ansi_reset)
                return False

    def detection_board(self, frame):
        results = self.model(frame)

        # convert bounding boxes into board coordinates
        self.reset_board()
        annotator = Annotator(frame)
        update_frame = np.copy(self.calibration_frame)
        for res in results:
            boxes = res.boxes
            for box in boxes:
                # (left, top, right, bottom) format
                shift_x = (-box.xyxy[0, 0] + box.xyxy[0, 2]) / 2
                shift_y = (-box.xyxy[0, 1] + box.xyxy[0, 3]) / 2
                x_c = box.xyxy[0, 0] + shift_x
                y_c = box.xyxy[0, 1] + shift_y

                self.update_board(np.float32([x_c, y_c]), box.cls)

                annotator.box_label(box.xyxy[0], self.model.names[int(box.cls)])
                update_frame = self.pose_estimator.draw_point(update_frame, np.float32([x_c, y_c]))

        frame = annotator.result()
        update_frame = self.pose_estimator.draw_cells(update_frame)

        return frame, update_frame

    def reset_board(self):
        for i in range(8):
            for j in range(8):
                self.matrix[i][j] = '---'

    def update_board(self, pixel, piece_class):
        # piece_class: 1, black (player); 2, black king; 3, white (computer); 4, white king
        try:
            i, j = self.pose_estimator.find_nearest_cell(pixel)
        except ValueError as e:
            i, j = 0, 0
            print(f"[VISION] Warning: At pixel {pixel}, detected piece class {piece_class}!!!")
            return

        # TODO: l'inizializzazione del reference frame della board non è consistente, perché non è consistente
        # findchessboard corners.... per ora non so come risolvere la cosa
        if piece_class == 1:
            self.matrix[j][7-i] = f'b{j}{7-i}'
        elif piece_class == 2:
            self.matrix[j][7-i] = f'B{j}{7-i}'
        elif piece_class == 3:
            self.matrix[j][7-i] = f'c{j}{7-i}'
        elif piece_class == 4:
            self.matrix[j][7-i] = f'C{j}{7-i}'
        else:
            raise ValueError(f"[VISION] ERROR: Invalid piece class {piece_class} for piece detected in {pixel}!")

    def find_board_differences(self, ignore_indeces):
        differences = []
        for i in range(len(self.old_matrix)):
            for j in range(len(self.old_matrix[0])):

                if (i, j) in ignore_indeces:
                    continue

                if self.old_matrix[i][j] != self.matrix[i][j]:
                    differences.append((i, j))
        return differences

    def convert_move_for_robot(self, move):
        # conversion from internal checkers reference system to pose_estimator reference system
        move_dict = {'from': (move[1] + 7, move[0]),
                     'to': (move[3] + 7, move[2])}

        # check for captures
        differences = self.find_board_differences(ignore_indeces=[move_dict['from'], move_dict['to']])
        if len(differences) == 1:
            move_dict['capture'] = differences[0]

        # check for king promotion
        if 'C' in self.matrix[move[2]][move[3]]:
            move_dict['promotion'] = True

        # check for king movement
        if 'C' in self.matrix[move[0]][move[1]]:
            move_dict['king'] = True

        return move_dict

    def play(self):
        self.print_matrix()
        print(ansi_cyan + "##### WELCOME TO CHECKERS ####" + ansi_reset)
        print("\nSome basic rules:")
        print("1.You enter the coordinates in the form i,j.")
        print("2.You can quit the game at any time by pressing enter.")
        print("3.You can surrender at any time by pressing 's'.")
        print("Now that you've familiarized yourself with the rules, enjoy!")
        while True:
            answer = input("\nFirst, we need to know, is jumping mandatory?[Y/n]: ")
            if answer == "Y" or answer == "y":
                self.mandatory_jumping = True
                break
            elif answer == "N" or answer == "n":
                self.mandatory_jumping = False
                break
            elif answer == "":
                print(ansi_cyan + "Game ended!" + ansi_reset)
                exit()
            elif answer == "s":
                print(ansi_cyan + "You've surrendered before the game even started.\nPathetic." + ansi_reset)
                exit()
            else:
                print(ansi_red + "Illegal input!" + ansi_reset)

        print("[MAIN] Initialization of the game directory...")
        game_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.game_directory = f"./game_{game_timestamp}/"
        os.mkdir(self.game_directory)

        print("[MAIN] Initialization of vision system...")
        print("\tPlease position the board. It has to be empty! (And don't move it during the game)")
        print("Press [m] in the visualization window to confirm the board position.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Display the resulting frame
            key = cv.waitKey(5)
            if key == 27:  # esc character
                exit()
            elif key == ord('m'):
                self.calibration_frame = frame
                cv.imwrite(self.game_directory + 'setup_frame.png', self.calibration_frame)
                break
            cv.imshow('Set-up Frame', frame)
        cv.destroyWindow('Set-up Frame')

        print("[MAIN] Setting pose-estimator parameters...")
        gray = cv.cvtColor(self.calibration_frame, cv.COLOR_BGR2GRAY)
        self.pose_estimator.estimate_parameters(gray)

        print("[MAIN] Now you can position the pieces on the board."
              "Press [m] in the visualization window to confirm the pieces start position.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Display the resulting frame
            key = cv.waitKey(5)
            if key == 27:  # esc character
                exit()
            elif key == ord('m'):
                self.detection_board(frame)
                cv.imwrite(self.game_directory + 'game_init.png', frame)
                break
            self.pose_estimator.draw_axis(frame)
            cv.imshow('Pieces Set-up Frame', frame)
        cv.destroyWindow('Pieces Set-up Frame')

        print("[MAIN] Robot set up...")
        self.robot = RoboticArm(link1=5, link2=5, cells=self.pose_estimator.get_cells())

        print("[MAIN] You can set-up the board. When you are ready, make your move.")

        while True:
            ret, frame = self.cap.read()

            if not ret:
                print("[MAIN] Can't receive frame (stream end?). Exiting ...")

            self.print_matrix()

            # player turn
            if self.player_turn is True:
                print(ansi_cyan + "\nPlayer's turn." + ansi_reset)

                if not self.has_player_move():
                    break

                if not self.wait_for_player_move():
                    # player wants to exit the game
                    break

            # computer turn
            else:
                print(ansi_cyan + "Computer's turn." + ansi_reset)
                print("[MAIN] Thinking...")

                self.old_matrix = self.matrix
                move = self.evaluate_states()

                # compute coordinates
                move_dict = self.convert_move_for_robot(move)
                print(move_dict)

                # robot actuation of the move
                print("[MAIN] Moving robot...")
                self.robot.move(move_dict)

            # check end-game conditions
            if self.player_pieces == 0:
                self.print_matrix()
                print(ansi_red + "You have no pieces left.\nYOU LOSE!" + ansi_reset)
                break
            elif self.computer_pieces == 0:
                self.print_matrix()
                print(ansi_green + "Computer has no pieces left.\nYOU WIN!" + ansi_reset)
                break
            elif self.computer_pieces - self.player_pieces == 7:
                wish = input("You have 7 pieces fewer than your opponent.Do you want to surrender?")
                if wish == "" or wish == "yes":
                    print(ansi_cyan + "Coward." + ansi_reset)
                    break
            self.player_turn = not self.player_turn

            # # visualization
            # cv.imshow('Game', frame)
            # if cv.waitKey(0) == ord('s'):
            #     print(ansi_cyan + "Coward." + ansi_reset)

            self.counter_picture += 1

        print("[MAIN] Game ended. Releasing resources...")
        self.cap.release()
        cv.destroyAllWindows()

        print("[MAIN] Game ended. Bye!")


if __name__ == '__main__':
    checkers = Checkers()
    checkers.play()
