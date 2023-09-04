import random
import sys
from read import readInput
from write import writeOutput
from host import GO
import copy

class MyPlayer():
    def __init__(self):
        self.type = 'minmax'

    def first_step(self,piece_type,possible_placements):
        mid = (2,2)
        # play black, place (2,2) first
        if piece_type==1 and len(possible_placements)==25: 
            return mid 
        # play white, if (2,2) is empty, place(2,2), else (3,2)
        if piece_type==2 and len(possible_placements)==24: #white
            if mid in possible_placements:
                return mid  
            else:
                return (3,2) 
    
    def action(self,possible_placements):
        if not possible_placements:
            return "PASS"
        else:
            return random.choice(possible_placements)

# copy from random_pkayer.py
def get_possible_placements(board, go, piece_type):  
    test_go = go.copy_board()
    test_go.update_board(board)
    possible_placements = []
    for i in range(N):
        for j in range(N):
            if test_go.valid_place_check(i, j, piece_type, test_check = True):
                possible_placements.append((i,j))
    return possible_placements

# copy from host.py 
def detect_neighbors(board,i,j): 
    neighbors = []
    # Detect borders and add neighbor coordinates
    if i > 0: neighbors.append((i-1, j))
    if i < len(board) - 1: neighbors.append((i+1, j))
    if j > 0: neighbors.append((i, j-1))
    if j < len(board) - 1: neighbors.append((i, j+1))
    return neighbors

def detect_neighbors_group(board, i,j): 
    neighbors_in_one_group = []
    # go through neighbors of a stone
    for stone in detect_neighbors(board, i, j):
        # if it's in one group, add to group neighbors
        if board[stone[0]][stone[1]] == board[i][j]:
            neighbors_in_one_group.append(stone)
    return neighbors_in_one_group

def detect_group(board, i, j):
    stack = [(i, j)]
    group_stones = []
    while stack:
        stone = stack.pop()
        group_stones.append(stone)
        # if group nieghbors not empty, add them to group
        for neighbor in detect_neighbors_group(board, stone[0], stone[1]):
            if neighbor not in stack and neighbor not in group_stones:
                stack.append(neighbor)
    return group_stones

def group_liberty(board, i, j):
    liberty = 0
    group_stones = detect_group(board, i, j)
    # go through stones in a group
    for stone in group_stones:
        # go through stone's neighbors
        for neighbor in detect_neighbors(board, stone[0],stone[1]):
            if board[neighbor[0]][neighbor[1]] == 0:
                liberty += 1
    return liberty

def detect_dead(board, piece_type):
    dead_stones = []
    # go through the board
    for i in range(N):
        for j in range(N):
            # consider stone with no (group) liberty as dead stone
            if board[i][j] == piece_type:
                if group_liberty(board, i, j)==0 and (i,j) not in dead_stones:
                    dead_stones.append((i, j))
    return dead_stones


def board_after_rmv(board, target):
    for stone in target:
        board[stone[0]][stone[1]] = 0
    return board

def board_after_rmv_dead(board, piece_type):
    dead_stones = detect_dead(board, piece_type)
    if not dead_stones:
        return board
    new_board = board_after_rmv(board, dead_stones)
    return new_board

def place_stone(board,placement,piece_type):
    # place stone,  and update board
    board_copy = copy.deepcopy(board)
    board_copy[placement[0]][placement[1]] = piece_type
    # update board after removing opponent's dead stone
    board_copy = board_after_rmv_dead(board_copy, 3-piece_type)
    return board_copy

def heuristic(board, next_player):
    me, opponent, heur_me, heur_opponent = 0, 0, 0, 0
    # go through board
    for i in range(N):
        for j in range(N):
            # if it's my stone
            if board[i][j] == my_type:
                me += 1
                heur_me += (me + group_liberty(board, i, j))
            # if it's opponet's stone
            elif board[i][j] == 3-my_type:
                opponent += 1
                heur_opponent += (opponent + group_liberty(board, i, j))
    # who plays next
    if next_player == my_type:
        return heur_me-heur_opponent
    return heur_opponent-heur_me

def minmax(board,max_depth,alpha,beta,ini_heur,next_player):
    # iteratively call minmax till max_depth==0
    if max_depth == 0:
        return ini_heur
    best = ini_heur
    possible_placements = get_possible_placements(board,go,next_player)
    # go through all possible placements 
    for placement in possible_placements:
        # place stone,  and update board
        next_board = place_stone(board,placement,next_player)
        # get heuristic of next_board
        heur_score = heuristic(next_board, 3-next_player)
        # iteratively call minmax
        minmax_evaluation = minmax(next_board,max_depth-1,alpha, beta,heur_score,3-my_type)
        current_score = -1 * minmax_evaluation
        if current_score > best:
            best = current_score
        new_score = -1 * best
        # Alpha beta pruning
        # if next player is opponent
        if next_player == 3-my_type:
            player = new_score
            # check prune
            if player < alpha:
                return best
            # if don't prune, update beta
            if best > beta:
                beta = best
        # if next player is me
        elif next_player == my_type:
            opponent = new_score
            # check prune
            if opponent < beta:
                return best
            # if don't prune, update alpha
            if best > alpha:
                alpha = best
    return best
    
def minmax_actions(board,max_depth, alpha, beta, piece_type):
    actions = []
    best = 0
    possible_placements = get_possible_placements(board,go,piece_type)
    # go through all possible placements 
    for placement in possible_placements:
        next_board = place_stone(board,placement,piece_type)
        # get heuristic of next_board
        heur_score = heuristic(next_board, 3-piece_type)
        # call minmax
        minmax_evaluation = minmax(next_board,max_depth,alpha, beta,heur_score,3-piece_type)
        current_score = -1 * minmax_evaluation
        # if actions is empty or if we have new best actions, replace actions
        if not actions or current_score > best:
            best = current_score
            alpha = best
            actions = [placement]
        # if we have action as good as the best one, we add it to actions
        elif current_score == best:
            actions.append(placement)
    return actions

def min(board,max_depth,alpha, beta,next_player):
    best = heuristic(board,next_player)
    # iteratively call minmax till max_depth==0
    if max_depth == 0:
        return best
    possible_placements = get_possible_placements(board,go,next_player)
    # go through all possible placements 
    for placement in possible_placements:
        next_board = place_stone(board,placement,next_player)
        max_evaluation = max(next_board, max_depth-1, alpha, beta, 3-next_player)
        curr_score = -1 * max_evaluation
        # update best if current score is higher
        if curr_score > best:
            best = curr_score
        player = -1 * best
        # prune
        if player < alpha:
            return best
        # update beta value
        if best > beta:
            beta = best
    return best

def max(board,max_depth,alpha, beta,next_player):
    best = heuristic(board,next_player)
    # iteratively call minmax till max_depth==0
    if max_depth == 0:
        return best
    possible_placements = get_possible_placements(board,go,next_player)
    # go through all possible placements 
    for placement in possible_placements:
        next_board = place_stone(board,placement,next_player)
        min_evaluation = min(next_board, max_depth-1, alpha, beta, 3-next_player)
        curr_score = -1 * min_evaluation
        # update best if current score is higher
        if curr_score > best:
            best = curr_score
        opponent = -1 * best
        # prune
        if opponent < beta:
            return best
        # update alpha value
        if best > alpha:
            beta = alpha
    return best

def minmax_actions_2(board,max_depth, alpha, beta, piece_type):
    actions = []
    best = 0
    possible_placements = get_possible_placements(board,go,piece_type)
    # go through all possible placements 
    for placement in possible_placements:
        next_board = place_stone(board,placement,piece_type)
        # get heuristic of next_board
        # heur_score = heuristic(next_board, 3-piece_type)
        # iteratively call min for opponent and call max in min for me
        min_evaluation = min(next_board,max_depth,alpha, beta,3-my_type)
        current_score = -1 * min_evaluation
        # if actions is empty or if we have new best actions, replace actions
        if not actions or current_score > best:
            best = current_score
            alpha = best
            actions = [placement]
        # if we have action as good as the best one, we add it to actions
        elif current_score == best:
            actions.append(placement)
    return actions

if __name__ == "__main__":
    N = 5
    my_type, previous_board, board = readInput(N,"input.txt")
    go = GO(N)
    go.set_board(my_type, previous_board, board)
    player = MyPlayer()
    # first step
    possible_placements = get_possible_placements(board,go, my_type)
    action = player.first_step(my_type,possible_placements)
    # second and after steps call minmax
    # if len(possible_placements) >0:
    #     max_depth = 3
    # else:
    #     max_depth = 5
    max_depth = 3
    ini_alpha = -10000
    ini_beta = 10000
    if not action:
        actions = minmax_actions_2(board,max_depth,ini_alpha,ini_beta,my_type)
        action = player.action(actions)
    writeOutput(action)