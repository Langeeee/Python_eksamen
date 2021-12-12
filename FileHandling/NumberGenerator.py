import numpy as np

class NumberGenerator:
    def __init__(self):
        print()

    def generateSequence(self, length):
        numbers = [int(x) for x in str(length)]
        returnVal = ""
        i = 0
        while(i<4): 
            returnVal += str(np.random.choice(numbers))
            i += 1
            
        return returnVal

