import sys; args=sys.argv[1:] # m, n
import os, time, math, random as rd
import tensorflow as tf
import keras
from keras import layers
import numpy as np
from collections import deque

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BOARD, EDGES, TAKEN, OWNERS, BOXES, VISITED = {}, set(), {}, {}, {}, []

def heuristic():
    moves = list(findMoves())
    safe = []
    for move in moves:
        creates_danger = False

        for box, edges in BOARD.items():
            if move in edges:
                taken = sum(1 for e in edges if e in TAKEN or e == move)
                if taken == 3:
                    creates_danger = True
                    break

        if not creates_danger:
            safe.append(move)

    if safe:
        return rd.choice(safe)
    else:
        return rd.choice(moves)
    
def random():
    return rd.choice(list(findMoves()))

def playGames(m, n, numOfGames):
    global TAKEN, OWNERS, BOARD, BOXES, EDGES
    p1Wins = 0
    p2Wins = 0
    box_cnt = [0,0]
    times = [0,0]
    moves = [0,0]
    for _ in range(1, numOfGames + 1):
        BOARD = {}
        EDGES = set() 
        TAKEN = {}   
        OWNERS = {}
        BOXES = {}
        VISITED = []
        BOXES, BOARD, adjMat, EDGES = initialzeBoard(n, m)
        OWNERS = {p : '' for p in range(len(BOXES))}

        players = ['p1', 'p2']
        boxes = {'p1': [], 'p2': []}
        turn = 0
        model = keras.models.load_model('model.keras', custom_objects={"GNNLayer": GNNLayer})
        adjMat = np.array(initialzeBoard(n, m)[2], dtype=np.float32)

        while findMoves():
            # printBoard(n, m, adjMat)
            player = players[turn % 2]
            
            if player == 'p1':
                model_time = time.time()
                state = np.array(makeNodeFeatMat(BOARD, TAKEN, n, m), dtype=np.float32)
                q_vals = model.predict([state[np.newaxis, :], adjMat[np.newaxis, :, :]], verbose=0)[0]
                q_vals[list(TAKEN.keys())] = -1e9 
                move = np.argmax(q_vals)
                model_time = round(time.time()-model_time, 3)
                times[0] += model_time
                moves[0] +=1
                
            else:
                model_time = time.time()
                # move = random()
                # move = heuristic()
                # move = getBestMoveAlphabeta((TAKEN, OWNERS), depth=3, maximizing_player=True)
                model_time = round(time.time()-model_time, 3)
                times[1] += model_time
                moves[1] +=1

            TAKEN[move] = player
            # print(f'{player} moves to {move}')
            
            scored = False
            for boxID, edges in BOXES.items():
                if OWNERS[boxID] == '' and all(e in TAKEN for e in edges):
                    OWNERS[boxID] = "x" if player == 'p1' else "o"
                    boxes[player].append(boxID)
                    if player == 'p1': box_cnt[0] += 1
                    else: box_cnt[1] += 1
                    scored = True
            
            if not scored:
                turn += 1
        p1Score = len(boxes['p1'])
        p2Score = len(boxes['p2'])
        if p1Score > p2Score:
            p1Wins += 1
        elif p2Score > p1Score:
            p2Wins += 1
        if _ % 1 == 0:
            print('game ' + str(_) + ' done')
            print(f"p1: {p1Wins}\np2: {p2Wins}")
            print(f'boxes:{box_cnt[0]/(_*n*m)}, {box_cnt[1]/(_*n*m)}')
            print(f'time/move: {times[0]/moves[0]}, {times[1]/moves[0]}\n')

def findMoves():
    return EDGES - TAKEN.keys()

def initialzeBoard(n, m):
    temp = {}
    horiz_count = (n+1)*m

    for r in range(n):
        for c in range(m):
            box_id = r*m + c

            top = r*m + c
            bottom = (r+1)*m + c
            left = horiz_count + r*(m+1) + c
            right = horiz_count + r*(m+1) + (c+1)

            temp[box_id] = [top, bottom, left, right]

    numberOfEdges = 2 * n * m + n + m
    edgeRep = {}

    for e in range(numberOfEdges):
        edgeRep[e] = []
        for box in range(n * m):
            if e in temp[box]: 
                edgeRep[e].append(box)

    adjMat = [[0 for _ in range(numberOfEdges)] for _ in range(numberOfEdges)]

    for boxID in range(n*m):
        edges = temp[boxID]
        for e1 in edges:
            for e2 in edges:
                adjMat[e1][e2] = 1
                adjMat[e2][e1] = 1
    return temp, edgeRep, adjMat, {e for e in range(numberOfEdges)}

def printBoard(n, m, adjMat):
    out = ""
    hCount = (n+1)*m

    for r in range(n):
        for c in range(m):
            out += "."
            edge = r*m + c
            out += "---" if edge in TAKEN else "   "
        out += ".\n"

        for c in range(m):
            leftEdge = hCount + r*(m+1) + c
            out += "|" if leftEdge in TAKEN else " "

            boxID = r*m + c
            out += f" {OWNERS[boxID]} " if OWNERS[boxID] else '   '

        rightEdge = hCount + r*(m+1) + m
        out += "|" if rightEdge in TAKEN else " "
        out += "\n"

    for c in range(m):
        out += "."
        edge = n*m + c
        out += "---" if edge in TAKEN else "   "
    out += ".\n"

    print(out,'\n')

    # for l in adjMat:
    #     print(l)

def printStats():
    global OWNERS
    p1 = p2 = 0
    print(OWNERS)
    for boxID, player in OWNERS.items():
        if player == 'x': p1+=1
        else: p2+=1 
    
    print(f'Percent of boxes owned by player 1: {(p1/(p1 + p2))*100}%')
    print(f'Percent of boxes owned by player 2: {(p2/(p1 + p2))*100}%\n')
    if p1 > p2:
        print('Player 1 wins')
    elif p2 > p1:
        print('Player 2 wins')
    else:
        print('Tied')

def makeNodeFeatMat(board, taken, n, m):
    temp = [[1 if i in taken else 0] for i in range(len(board))]
    # print(len(board))

    for i in range(len(board)):
        if i < m or (i >= (n + 1) * m - m and i < (n + 1) * m) or (i >= (n + 1) * m and ((i - (m * (n+1))) % (m + 1) == 0 or (i - (m * (n+1))) % (m + 1) == n)):
            temp[i].append(1)
        else: temp[i].append(0)
    # print(temp)
    return temp

class GNNLayer(layers.Layer):
    def __init__(self, units=64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = layers.Dense(units, activation='relu')

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

    def call(self, x, adj):
        neighbor_info = tf.matmul(adj, x)
        return self.dense(neighbor_info)
    
def buildGNNModel(features):
    node_input = layers.Input(shape=(None, features))
    adj_input = layers.Input(shape=(None, None))
    
    x = GNNLayer(64)(node_input, adj_input)
    x = GNNLayer(64)(x, adj_input)
    
    q_values = layers.Dense(1)(x) 
    q_values = layers.Reshape((-1,))(q_values)
    
    model = keras.Model(inputs=[node_input, adj_input], outputs=q_values)
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse')
    return model

def trainAgent(episodes, n, m, adjMat, nodeFeatMat): # from Hugging Face
    global TAKEN
    model = buildGNNModel(len(nodeFeatMat[0]))
    target_model = buildGNNModel(len(nodeFeatMat[0]))
    target_model.set_weights(model.get_weights())
    
    epsilon, gamma, batchSize = 1.0, 0.99, 32
    memory = deque(maxlen=2000)
    adjMat = np.array([np.array(a) for a in adjMat])

    for ep in range(episodes):
        global TAKEN, OWNERS
        TAKEN = {}; OWNERS = {i: '' for i in range(len(BOXES))}
        
        state = np.array(makeNodeFeatMat(BOARD, TAKEN, n, m), dtype=np.float32)
        done = False
        
        while not done:
            if rd.random() < epsilon:
                action = rd.choice(list(findMoves()))
            else:
                qVals = model.predict([state[np.newaxis, :], adjMat[np.newaxis, :]], verbose=0)[0]
                qVals[list(TAKEN.keys())] = -1e9
                action = np.argmax(qVals)
            
            nextState, reward, done = step(action, n, m)
            
            if reward == 0 and not done:
                opponent_turn = True
                
                while opponent_turn and not done:
                    opp_state = np.array(makeNodeFeatMat(BOARD, TAKEN, n, m), dtype=np.float32)
                    opp_qVals = model.predict([opp_state[np.newaxis, :], adjMat[np.newaxis, :]], verbose=0)[0]
                    opp_qVals[list(TAKEN.keys())] = -1e9
                    opp_action = np.argmax(opp_qVals)
                        
                    TAKEN[opp_action] = 'o'
                    
                    opp_scored = False
                    for boxID, edges in BOXES.items():
                        if OWNERS[boxID] == '' and all(e in TAKEN for e in edges):
                            OWNERS[boxID] = 'o'
                            reward -= 1.0 
                            opp_scored = True
                    
                    done = len(TAKEN) == len(EDGES)
                    if not opp_scored:
                        opponent_turn = False
                
                nextState = np.array(makeNodeFeatMat(BOARD, TAKEN, n, m), dtype=np.float32)

            memory.append((state, action, reward, nextState, done))
            state = nextState
            
            if len(memory) > batchSize:
                batch = rd.sample(memory, batchSize)
                states = np.array([b[0] for b in batch])
                actions = np.array([b[1] for b in batch])
                rewards = np.array([b[2] for b in batch])
                nextStates = np.array([b[3] for b in batch])
                dones = np.array([b[4] for b in batch])
                
                # Bellman Update
                targetQ = model.predict([states, np.tile(adjMat, (batchSize, 1, 1))], verbose=0)
                nextQ = target_model.predict([nextStates, np.tile(adjMat, (batchSize, 1, 1))], verbose=0)

                for i in range(batchSize):
                    if dones[i]:
                        targetQ[i][actions[i]] = rewards[i]
                    else:
                        targetQ[i][actions[i]] = rewards[i] + gamma * np.max(nextQ[i])
                
                model.train_on_batch([states, np.tile(adjMat, (batchSize, 1, 1))], targetQ)
        
        epsilon *= 0.99
        if ep % 10 == 0: 
            target_model.set_weights(model.get_weights())
        print(f"Episode {ep} done, Epsilon: {round(epsilon, 2)}")
            
    model.save("model.keras")

def step(action, n, m):
    global TAKEN, OWNERS
    if action in TAKEN:
        return None, -10.0, True # illegal move

    TAKEN[action] = 'x'
    
    reward = 0
    for boxID, edges in BOARD.items():
        if boxID in OWNERS and OWNERS[boxID] == '' and all(e in TAKEN for e in edges):
            OWNERS[boxID] = 'x'
            reward += 1
            
    done = len(TAKEN) == len(EDGES)

    newFeatures = np.array([[1 if i in TAKEN else 0, makeNodeFeatMat(BOARD, TAKEN, n, m)[i][1]] for i in range(len(EDGES))])
    return newFeatures, reward, done

def alphabeta(node, depth, alpha, beta, maximizing_player): #from GeeksforGeeks
    global VISITED

    if depth == 0 or boardEval(node[1]) != 0:
        return boardEval(node[1])

    if maximizing_player == 'x':
        max_eval = float('-inf')
        for (child, edge) in getChildren(node, maximizing_player):
            VISITED.append((node, child, "visit"))
            val = alphabeta(child, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, val)
            alpha = max(alpha, val)
            if beta <= alpha:
                VISITED.append((node, child, "cut"))
                break
        return max_eval

    else:
        min_eval = float('inf')
        for (child, edge) in getChildren(node, maximizing_player):
            VISITED.append((node, child, "visit"))
            val = alphabeta(child, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, val)
            beta = min(beta, val)
            if beta <= alpha:
                VISITED.append((node, child, "cut"))
                break
        return min_eval

def getBestMoveAlphabeta(root, depth, maximizing_player):
    best_moves =[]
    
    if maximizing_player:
        best_score = float('-inf')
        for child, edge in getChildren(root, 'x'):
            score = alphabeta(child, depth - 1, float('-inf'), float('inf'), False)
            if score > best_score:
                best_score = score
                best_moves = [edge] 
            elif score == best_score:
                best_moves.append(edge) 
                
    else: 
        best_score = float('inf')
        for child, edge in getChildren(root, 'o'):
            score = alphabeta(child, depth - 1, float('-inf'), float('inf'), True)
            if score < best_score:
                best_score = score
                best_moves =[edge]
            elif score == best_score:
                best_moves.append(edge)
    return rd.choice(best_moves)

def getChildren(node, player): #from GeeksforGeeks
    taken, owners = node
    children = []
    moves = findMoves()
    for edge in moves:
        new_taken = {k: v for k, v in taken.items()}
        new_owners = {k: v for k, v in owners.items()}
        
        new_taken[edge] = player
        
        for boxID, edges in BOXES.items():
            if new_owners[boxID] == '': 
                if all(e in new_taken for e in edges):
                    new_owners[boxID] = player 
                    
        child_node = (taken, owners)
        children.append((child_node, edge))
        
    return children

def boardEval(owners):
    return sum(1 if owner == 'x' else -1 if owner == 'o' else 0 for boxID, owner in owners.items())

def main():
    global BOARD, EDGES, OWNERS, BOXES
    m, n = int(args[0]), int(args[1])
    BOXES, BOARD, adjMat, EDGES = initialzeBoard(m, n)  
    OWNERS = {n : '' for n in range(len(BOXES))}
    nodeFeatMat = makeNodeFeatMat(BOARD, TAKEN, m, n)
    p1 = {'moves': findMoves()} 
    p2 = {'moves': findMoves()} 
    
    trainAgent(400, m, n, adjMat, nodeFeatMat)
    # playGames(n, m, 10)
    # printBoard(n,m, adjMat)
    # printStats()

if __name__ == '__main__': main()
