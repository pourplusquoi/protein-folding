''' protein-folding

Protein-folding problem requires to find the folding structure of a given 
protein sequence with minimum free energy.

Protein-folding problem is an NP problem. Although the algorithm makes use 
of symmetry to reduce search space and avoids repetitive visit to the same 
structure, the algorithm still needs O(3^n) time and O(3^n) space in the 
worst case. Besides, to further reduce search space, the algorithm only 
consider the folding after which the free energy of structure becomes lower. 
This also lead to the consequence that global minimum might not be found by 
the algorithm.

The algorithm is, from initial structure, consider those folding actions 
that decrease free energy, push these strctures in a min-heap, with free 
energy as priority. Each time select the state at the top of min-heap and 
expand. Repeat until min-heap becomes empty or time is up. Return the 
folding structure with lowest free energy among all of visited structures.

Note that the lowest free energy in record decreases monotonically, and it 
has lower bound. Therefore, it would guarantee to converge on some positive 
value, if the time were unlimited.'''

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
    # self.protein: the protein that problem is dealing with
    def __init__(self, protein):
        self.protein = protein

    # Determine the start state of search.
    def startEnergyAndState(self):
        zero, one = np.complex(0, 0), np.complex(1, 0)
        initState = np.concatenate(([zero], 
            np.array([one for i in range(len(self.protein.sequence))])))

        return (Protein(self.protein.sequence, initState).calcEnergy(), 
            self.encode(initState))

    # Determine the successor states and the free energy of successor states.
    def succAndEnergy(self, energyAndState, visited):
        succAndEnergyArray = []
        energy, state = energyAndState[0], self.decode(energyAndState[1])

        for i in range(2, len(self.protein.sequence)):
            for j in np.array([1, np.complex(0, 1), -1, np.complex(0, -1)]):
                if j == state[i] or j == -state[i - 1]:
                    continue

                newState = np.copy(state)
                newState[i] = j
                newStateNum = self.encode(newState)
                
                # avoid revisit
                if newStateNum in visited:
                    continue
                
                visited.add(newStateNum)
                newEnergy = Protein(self.protein.sequence, newState).calcEnergy()
                if newEnergy is not None and newEnergy < energy:
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
        for i in range(1, len(self.protein.sequence)):
            array.append(num % 4)
            num /= 4

        array = np.array(array)
        return (array == 0)*1 + (array == 1)*np.complex(0, 1) + \
            (array == 2)*(-1) + (array == 3)*np.complex(0, -1)

    # Solve the protein-folding problem with algorithm provided above
    def solve(self):
        pq = []
        minEnergyAndState = self.startEnergyAndState()
        visited = {minEnergyAndState[1]}
        heapq.heappush(pq, minEnergyAndState)

        while len(pq) > 0:
            current = heapq.heappop(pq)

            if current[0] < minEnergyAndState[0]:
                print current[0], '\n', Protein(self.protein.sequence, 
                    self.decode(current[1])).toString(), '\n', \
                    '------------------------------'
                minEnergyAndState = current
            
            children = self.succAndEnergy(current, visited)
            if children is not None:
                for child in children:
                    heapq.heappush(pq, child)
        
        return minEnergyAndState

if __name__ == '__main__':
    protein = Protein(np.array([0,0,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,0,0]), 
        np.array([0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]))
    ProteinFoldingProblem(protein).solve()
