'''
The python code is used to solve protein-folding problem in Assignment 1 
of COMP 557.

Protein-folding problem requires to find the folding structure of a given 
protein sequence with minimum free energy.

Protein-folding problem is an NP problem. Although the algorithm makes use 
of symmetry to reduce search space and avoids repetitive visit to the same 
structure, the algorithm still needs O(3^n) time and O(3^n) space in the 
worst case. Besides, to further reduce search space, the algorithm only 
consider the folding after which the free energy of structure becomes lower. 
This also lead to the consequence that global minimum might not be found by 
the algorithm.

Greedy Algorithm:
The algorithm is, from initial structure, consider those folding actions 
that decrease free energy, push these strctures in a min-heap, with free 
energy as priority. Each time select the state at the top of min-heap and 
expand. Repeat until min-heap becomes empty or time is up. Return the 
folding structure with lowest free energy among all of visited structures.

Radom Greedy Algorithm:
Similar to Greedy Algorithm, with extra hyper-parameter 'self.numChild' 
controlling maximum number od accepted children. At each expand, randomly 
select successors with constraint on number. A min-heap is maintained here 
to provide the most promising candidate each time.

Simulated Annealing Algorithm:
The probability of making a downhill move is higher when T is higher, and 
is lower as T comes down. At high temperatures simulated annealing performs 
a random walk, while at lower temperatures, it is a stochastic hill climber.

Genetic Algorithm:
The algorithm is based on the mechanics of natural selection. A mating pool 
with population upper bound is kept. Each time, every pair reproduce a 
descendent at random, two genes corss over at random position, and the new 
gene mutates with given probability. Each time, those folding structures 
with lower free energy are kept, others are abandoned.

Note that the lowest free energy in record decreases monotonically, and it 
has lower bound. Therefore, all of these algorithms guarantee that the 
lowest free energy would converge on some positive value, if the time were 
unlimited.'''

import heapq
import numpy as np

class Protein:
    # self.sequence: the acid sequence
    # self.relative: the coordinate of acid relative to the previoud acid
    #                on complex plane
    def __init__(self, sequence, relative):
        self.sequence = sequence
        self.relative = relative

    # Calculate the absolute coordinate of acid on complex plane.
    def calcAbsCoords(self):
        absolute = np.cumsum(self.relative)
        pointSet = set()
        for pt in absolute:
            if pt in pointSet:
                return None
            pointSet.add(pt)
        return absolute

    # Calculate the free energy of a protein with specific folding structure.
    def calcEnergy(self):
        absolute = self.calcAbsCoords()
        if absolute is None:
            return None

        hydro = []
        for i in range(len(self.sequence)):
            if self.sequence[i] == 1:
                hydro.append(absolute[i])

        hydro = np.array([hydro])
        numHydro = np.sum(self.sequence)
        return np.sum(np.abs(np.repeat(hydro, numHydro, 0) \
            - np.repeat(hydro.T, numHydro, 1))) / 2

    # Convert the protein folding structure into string.
    def toString(self):
        absolute = self.calcAbsCoords()
        real, imag = np.array(np.round(absolute.real), dtype = int), \
            np.array(np.round(absolute.imag), dtype = int)

        minX, maxX = np.min(real), np.max(real)
        minY, maxY = np.min(imag), np.max(imag)

        out = [[' ' for i in range((maxX - minX)*2 + 1)] \
            for j in range((maxY - minY)*2 + 1)]

        lastX, lastY = (real[0] - minX)*2, (imag[0] - minY)*2

        if self.sequence[0] == 0:
            out[lastY][lastX] = 'O'
        else:
            out[lastY][lastX] = 'X'

        for i in range(1, len(self.sequence)):
            currX, currY = (real[i] - minX)*2, (imag[i] - minY)*2
            if currX != lastX:
                out[currY][(currX + lastX) / 2] = '-'
            else:
                out[(currY + lastY) / 2][currX] = '|'
            if self.sequence[i] == 0:
                out[currY][currX] = 'O'
            else:
                out[currY][currX] = 'X'
            lastX, lastY = currX, currY

        string = ''
        for i in range(len(out)):
            for j in range(len(out[0])):
                string += out[i][j]
            string += '\n'

        return string

class ProteinFoldingProblem():
    # self.sequence: sequence of the protein that problem is dealing with
    def __init__(self, sequence):
        self.sequence = sequence

    # Determine the start state of search.
    def startEnergyAndState(self):
        zero, one = np.complex(0, 0), np.complex(1, 0)
        initState = np.concatenate(([zero], 
            np.array([one for i in range(len(self.sequence))])))

        return (Protein(self.sequence, initState).calcEnergy(), 
            self.encode(initState))

    # Determine the successor states and the free energy of successor states.
    def succAndEnergy(self, energyAndState, visited, option=0):
        succAndEnergyArray = []
        energy, state = energyAndState[0], self.decode(energyAndState[1])

        for i in range(2, len(self.sequence)):
            for j in np.array([1, np.complex(0, 1), -1, np.complex(0, -1)]):
                if j == state[i] or j == -state[i - 1] or \
                    (i + 1 < len(self.sequence) and j == -state[i + 1]):
                    continue

                newState = np.copy(state)
                newState[i] = j
                newStateNum = self.encode(newState)
                
                # avoid revisit
                if newStateNum in visited:
                    continue
                
                visited.add(newStateNum)
                newEnergy = Protein(self.sequence, newState).calcEnergy()
                if newEnergy is not None and (newEnergy < energy or option==1):
                    succAndEnergyArray.append((newEnergy, newStateNum))
        
        return succAndEnergyArray

    # A helper function to encode a state (array) into a integer.
    def encode(self, array):
        array = ((array == np.complex(0, 1))*1 + (array == -1)*2 + \
            (array == np.complex(0, -1))*3).tolist()
        array.reverse()

        num = 0
        for i in range(len(array) - 1):
            num = num * 4 + array[i]
        return num

    # A helper function to decode a integer back into state.
    def decode(self, num):
        array = [-1]
        for i in range(1, len(self.sequence)):
            array.append(num % 4)
            num /= 4

        array = np.array(array)
        return (array == 0)*1 + (array == 1)*np.complex(0, 1) + \
            (array == 2)*(-1) + (array == 3)*np.complex(0, -1)

    
class Algorithm():
    # This is an abstract class.
    # self.problem: the folding problem that needs solving
    # self.maxIter: the maximum iteration of algorithm
    def __init__(self, problem, maxIter):
        self.problem = problem
        self.maxIter = maxIter

    def solve(self):
        raise NotImplementedError("Override me")

class GreedyAlgorithm(Algorithm):
    def __init__(self, problem, maxIter=100000):
        Algorithm.__init__(self, problem, maxIter)

    def solve(self):
        minEnergyAndState = self.problem.startEnergyAndState()
        visited = {minEnergyAndState[1]}
        pq = [minEnergyAndState]

        iteration = 0
        while iteration < self.maxIter and len(pq) > 0:
            current = heapq.heappop(pq)

            if current[0] < minEnergyAndState[0]:
                print current[0], '\n', Protein(self.problem.sequence, 
                    self.problem.decode(current[1])).toString(), '\n', \
                    '------------------------------'
                minEnergyAndState = current
            
            children = self.problem.succAndEnergy(current, visited)
            if len(children) > 0:
                for child in children:
                    heapq.heappush(pq, child)
            
            iteration += 1

        return minEnergyAndState

class RandomGreedyAlgorithm(Algorithm):
    # self.numChild: the maximum number of selected children at each expand
    def __init__(self, problem, numChild=2, maxIter=100000):
        Algorithm.__init__(self, problem, maxIter)
        self.numChild = numChild

    def solve(self):
        minEnergyAndState = self.problem.startEnergyAndState()
        visited = {minEnergyAndState[1]}
        pq = [minEnergyAndState]

        iteration = 0
        while iteration < self.maxIter and len(pq) > 0:
            current = heapq.heappop(pq)

            if current[0] < minEnergyAndState[0]:
                print current[0], '\n', Protein(self.problem.sequence, 
                    self.problem.decode(current[1])).toString(), '\n', \
                    '------------------------------'
                minEnergyAndState = current
            
            children = self.problem.succAndEnergy(current, visited)
            if len(children) > 0:
                indices = np.unique(np.array(np.round(np.random.rand(
                    self.numChild)*(len(children) - 1)), dtype=int))
                for index in indices:
                    heapq.heappush(pq, children[index])

            iteration += 1

        return minEnergyAndState

class SimulatedAnnealingAlgorithm(Algorithm):
    def __init__(self, problem, maxIter=100000):
        Algorithm.__init__(self, problem, maxIter)

    def solve(self):
        def calcT(time):
            return 1.0 / time;

        minEnergyAndState = self.problem.startEnergyAndState()
        visited = {minEnergyAndState[1]}
        
        time = 1
        T = calcT(time)
        while time <= self.maxIter and T > 0:
            print minEnergyAndState[0], '\n', Protein(self.problem.sequence, 
                self.problem.decode(minEnergyAndState[1])).toString(), '\n', \
                '------------------------------'
            
            children = self.problem.succAndEnergy(
                minEnergyAndState, visited, 1)
            if len(children) > 0:
                index = np.int(np.random.rand()*len(children))
                diff = children[index][0] - minEnergyAndState[0]
                if diff < 0 or np.random.rand() < np.exp(-diff / T):
                    minEnergyAndState = children[index]
            else:
                break

            time += 1
            T = calcT(time)
        
        return minEnergyAndState

class GeneticAlgorithm(Algorithm):
    # self.population: the population upper bound of mating pool
    # self.mutationRate: the possibility that each entry of gene mutates
    def __init__(self, problem, population=10, mutationRate=0.1, \
            maxIter=100000):
        Algorithm.__init__(self, problem, maxIter)
        self.population = population
        self.mutationRate = mutationRate

    # Makes parent genes cross over at random position.
    def crossOver(self, stateA, stateB):
        coordA = self.problem.decode(stateA)
        coordB = self.problem.decode(stateB)
        pointIdx = np.int(np.random.rand()*len(coordA))

        temp = coordA.copy()
        coordA = np.concatenate((coordA[0: pointIdx], coordB[pointIdx:]))
        coordB = np.concatenate((coordB[0: pointIdx], temp[pointIdx:]))
        return coordA, coordB

    # Makes child gene mutate with given possibility.
    def mutate(self, coord):
        flip = np.random.rand(len(coord)) < self.mutationRate
        opts = [1, np.complex(0, 1), -1, np.complex(0, -1)]
        
        for i in range(1, len(flip)):
            if flip[i] == True:
                coord[i] = opts[np.int(np.random.rand()*4)]

        return self.problem.encode(coord)

    def solve(self):
        minEnergyAndState = self.problem.startEnergyAndState()
        visited = {minEnergyAndState[1]}
        pq = [minEnergyAndState, minEnergyAndState]

        for iteration in range(self.maxIter):
            for i in range(len(pq)):
                for j in range(i + 1, len(pq)):
                    stateA, stateB = self.crossOver(pq[i][1], pq[j][1])
                    newNums = [self.mutate(stateA), self.mutate(stateB)]
                    for newStateNum in newNums:

                        # avoid revisit
                        if newStateNum in visited:
                            continue

                        visited.add(newStateNum)
                        newState = self.problem.decode(newStateNum)
                        newEnergy = Protein(self.problem.sequence, 
                            newState).calcEnergy()

                        if newEnergy is not None:
                            heapq.heappush(pq, (newEnergy, newStateNum))

            if pq[0][0] < minEnergyAndState[0]:
                minEnergyAndState = pq[0]
                print minEnergyAndState[0], '\n', Protein(self.problem.sequence, 
                    self.problem.decode(minEnergyAndState[1])).toString(), \
                    '\n', '------------------------------'

            aux = []
            for i in range(min(len(pq), self.population)):
                heapq.heappush(aux, heapq.heappop(pq))
            pq = aux

        return minEnergyAndState

if __name__ == '__main__':
    sequence = np.array([0,0,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0])
    GeneticAlgorithm(ProteinFoldingProblem(sequence)).solve()
