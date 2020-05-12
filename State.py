import numpy as np

class State:

    def __init__(self, dim=None, state=None, densityMatrix=None):
        if(dim is None):
            raise ValueError('dim of state not defined')
        if(type(dim)!=int):
            raise ValueError('dim should be int')
        self.dim=dim

        if(densityMatrix is None):
            if(state is None):
                self.state=1
                for i in range(dim):
                    self.state=np.kron(self.state, [1,0])
                self.densityMatrix = np.dot(state, state.conjugate().reshape(1, 2 ** dim))
            else:
                if(type(state)==list):
                    state=np.array(state)
                    if(state.shape!=(2**dim, ) and state.shape!=(2**dim, 1)):
                        raise ValueError('state shape should be %s' % (str((2**dim,))))
                    self.state=state.reshape(2**dim,1)
                    self.densityMatrix=np.dot(state.reshape(2**dim,1), state.conjugate().reshape(1,2**dim))
                elif (type(state)==np.ndarray):
                    if(state.shape!=(2**dim,) and state.shape!=(2**dim,1)):
                        raise ValueError('state shape should be %s or %s' % (str((2 ** dim,)), str((2**dim,1))))
                    self.state=state.reshape(2**dim,1)
                    self.densityMatrix=np.dot(state.reshape(2**dim,1), state.conjugate().reshape(1,2**dim))
                else:
                    raise ValueError('state type error')
        else:
            if (type(densityMatrix) == list):
                densityMatrix = np.array(densityMatrix)
                if (densityMatrix.shape != (2 ** dim, 2 ** dim)):
                    raise ValueError('density matrix shape should be %s' % (str((2 ** dim, 2 ** dim))))
                self.densityMatrix = densityMatrix
            elif (type(densityMatrix) == np.ndarray):
                if (densityMatrix.shape != (2 ** dim, 2 ** dim)):
                    raise ValueError('density matrix shape should be %s' % (str((2 ** dim, 2 ** dim))))
                self.densityMatrix= densityMatrix
            else:
                raise ValueError('density matrix type error')


def randomState(dim=1):#old fashioned, need updating
    pa = np.random.rand(2)
    pa /= np.sqrt(np.sum(np.square(np.abs(pa))))
    pa = np.array([pa[0], pa[1]*np.exp(1j*np.random.random())]).reshape(2, 1)
    res=pa
    for i in range(dim-1):
        pa = np.random.rand(2)
        pa /= np.sqrt(np.sum(np.square(np.abs(pa))))
        pa = np.array([pa[0], pa[1]*np.exp(1j*np.random.random())]).reshape(2, 1)
        res=np.kron(res,pa)
    return res

def randomStateClass(dim=1):#return a class <State> variable
    pa = np.random.rand(2)
    pa /= np.sqrt(np.sum(np.square(np.abs(pa))))
    pa = np.array([pa[0], pa[1]*np.exp(1j*np.random.random())]).reshape(2, 1)
    res=pa
    for i in range(dim-1):
        pa = np.random.rand(2)
        pa /= np.sqrt(np.sum(np.square(np.abs(pa))))
        pa = np.array([pa[0], pa[1]*np.exp(1j*np.random.random())]).reshape(2, 1)
        res=np.kron(res,pa)
    resState=State(dim, res)
    return resState

def catState(dim=1):#return a cat state in <Class 'State'> form
    pa = np.zeros(2**dim, 1)
