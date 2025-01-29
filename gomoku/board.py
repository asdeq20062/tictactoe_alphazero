import copy
import random
import numpy as np
from constants import *


class Board:
    def __init__(self, board=None, current_player=None, available_moves=None):
        if board is None:
            # initialize empty board
            self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        else:
            self.board = board

        self.current_player = current_player or BLACK_PLAYER

        if available_moves is None:
            self.available_moves = set()
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    if self.board[i][j] == EMPTY:
                        self.available_moves.add((i, j))
        else:
            self.available_moves = available_moves

    def deepcopy(self):
        new_board = copy.deepcopy(self.board)
        new_available_moves = copy.deepcopy(self.available_moves)
        new_current_player = self.current_player
        return Board(new_board, new_current_player, new_available_moves)


    def print_board(self):
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                print("X" if self.board[row][col] == BLACK_PLAYER else "O" if self.board[row][col] == WHITE_PLAYER else ".", end=' ')
            print()

    def switch_player(self):
        self.current_player = WHITE_PLAYER if self.current_player == BLACK_PLAYER else BLACK_PLAYER

    def get_current_player(self):
        return self.current_player

    def get_last_player(self):
        return BLACK_PLAYER if self.current_player == WHITE_PLAYER else WHITE_PLAYER
    
    def get_player_move(self):
        while True:
            row = input("Enter row: ")
            col = input("Enter col: ")
            # check row and col is int
            if not row.isdigit() or not col.isdigit():
                print("Invalid input")
                continue
            if self.is_valid_move((int(row), int(col))):
                return (int(row), int(col))
            else:

                print("Invalid move")

    def is_valid_move(self, move):
        return move in self.available_moves
    
    def get_random_move(self):
        move = random.choice(list(self.available_moves))
        return move

    def move(self, move):
        row, col = move
        self.board[row][col] = self.current_player
        self.switch_player()
        self.available_moves.remove(move)

    def get_available_moves(self):
        return self.available_moves
    
    def get_board(self):
        return self.board
    
    def get_state(self):
        black_player_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        white_player_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        empty_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        current_player_board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if self.board[i][j] == BLACK_PLAYER:
                    black_player_board[i][j] = 1
                elif self.board[i][j] == WHITE_PLAYER:
                    white_player_board[i][j] = 1
                else:
                    empty_board[i][j] = 1

                if self.current_player == BLACK_PLAYER:
                    current_player_board[i][j] = 1
                else:
                    current_player_board[i][j] = 0
        return black_player_board, white_player_board, empty_board, current_player_board
    
    def check_game_over(self, move):
        """
        檢查五子棋是否有玩家獲勝或平局。

        :param board: 二維列表，表示棋盤 (N x N)，0 為空，1 和 2 分別代表兩位玩家
        :param x: 最近落子的位置 x 坐標 (行索引)
        :param y: 最近落子的位置 y 坐標 (列索引)
        :param player: 當前玩家 (1 或 2)
        :return: "winner" 如果該玩家獲勝，"draw" 如果平局，否則 "none"
        """
        if move is None:
            return WINNER_NONE
        
        row, col = move
        player = self.board[row][col]

        # check row
        min_col = max(0, col - (CONSECUTIVE_STONE_COUNT - 1))
        max_col = min(BOARD_SIZE - 1, col + CONSECUTIVE_STONE_COUNT)
        consecutive_count = 0
        for j in range(min_col, max_col + 1):
            if self.board[row][j] == player:
                consecutive_count += 1
            else:
                consecutive_count = 0
            if consecutive_count == CONSECUTIVE_STONE_COUNT:
                return player
            
        # check column
        min_row = max(0, row - (CONSECUTIVE_STONE_COUNT - 1))
        max_row = min(BOARD_SIZE - 1, row + CONSECUTIVE_STONE_COUNT)
        consecutive_count = 0
        for i in range(min_row, max_row + 1):
            if self.board[i][col] == player:
                consecutive_count += 1
            else:
                consecutive_count = 0
            if consecutive_count == CONSECUTIVE_STONE_COUNT:
                return player

        # check diagonal
        min_row = max(0, row - (CONSECUTIVE_STONE_COUNT - 1))
        min_col = max(0, col - (CONSECUTIVE_STONE_COUNT - 1))
        min_row_diff = row - min_row
        min_col_diff = col - min_col
        min_diff = min(min_row_diff, min_col_diff)
        min_row = row - min_diff
        min_col = col - min_diff

        max_row = min(BOARD_SIZE - 1, row + CONSECUTIVE_STONE_COUNT)
        max_col = min(BOARD_SIZE - 1, col + CONSECUTIVE_STONE_COUNT)
        max_row_diff = max_row - row
        max_col_diff = max_col - col
        max_diff = min(max_row_diff, max_col_diff)
        max_row = row + max_diff
        max_col = col + max_diff

        consecutive_count = 0
        while min_row <= max_row or min_col <= max_col:
            if self.board[min_row][min_col] == player:
                consecutive_count += 1
            else:
                consecutive_count = 0
            if consecutive_count == CONSECUTIVE_STONE_COUNT:
                return player
            min_row += 1
            min_col += 1

        # check anti-diagonal
        min_row = max(0, row - (CONSECUTIVE_STONE_COUNT - 1))
        max_col = min(BOARD_SIZE - 1, col + CONSECUTIVE_STONE_COUNT)
        min_row_diff = row - min_row
        max_col_diff = max_col - col
        min_diff = min(min_row_diff, max_col_diff)
        min_row = row - min_diff
        max_col = col + min_diff

        max_row = min(BOARD_SIZE - 1, row + CONSECUTIVE_STONE_COUNT)
        min_col = max(0, col - (CONSECUTIVE_STONE_COUNT - 1))
        max_row_diff = max_row - row
        min_col_diff = col - min_col
        min_diff = min(max_row_diff, min_col_diff)
        max_row = row + min_diff
        min_col = col - min_diff

        consecutive_count = 0
        while min_row <= max_row or max_col >= min_col:
            if self.board[min_row][max_col] == player:
                consecutive_count += 1
            else:
                consecutive_count = 0
            if consecutive_count == CONSECUTIVE_STONE_COUNT:
                return player
            min_row += 1
            max_col -= 1
    
        if len(self.available_moves) == 0:
            return WINNER_DRAW
        
        return WINNER_NONE