class romobject:
    def __init__(self, umat, senergymat, coeffsmat, mean_data):
#       pickle = __import__('pickle')
        self.umat = umat
        self.senergymat = senergymat
        self.coeffsmat = coeffsmat
        self.mean_data = mean_data
        
    #pickling them
    #def save(self, filename):
    #    with open(filename, 'wb') as output:
    #        pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
