from State import *
from Operation import *

class Circuit:
    """
     __init__, num_wires should be defined once a new object is generated
     wires: 0, 1, ..., num_wires-1
     op_queue: list of Operation Class
    """
    def __init__(self, num_wires=None, initial_state=None):
        if(num_wires is None):
            raise ValueError('Circuit num_wires not defined')
        if(type(num_wires) is not int):
            raise ValueError('Circuit num_wires should be int')
        self.num_wires=num_wires
        self.op_queue=[]
        if(initial_state is None):
            self.initial_state=None
        elif(type(initial_state)==State):
            self.initial_state=initial_state
        else:
            raise ValueError('initial state type error')

    def setInitialState(self, initial_state=None):
        if(initial_state is None):
            raise ValueError('initial state not defined')
        self.initial_state=initial_state
    """
    add(op_queue): add operator list or single operator
    """
    def add(self, op_queue=None):
        if(type(type(op_queue))==ABCMeta):
            #1 operation
            self.op_queue.append(op_queue)
        elif (type(op_queue)==list):
            for op in op_queue:
                if(type(type(op))!=ABCMeta):
                    raise ValueError('operation type should be class')
            self.op_queue.extend(op_queue)
    """
    (circuit, wires): add operator list of given circuit(circuit) to current circuit(self)
    wires:list of wires of self that will be extended. len(wires) must be equal to circuit.num_wires and no larger than self.num_wires
    """
    def extend(self, circuit, wires):
        if(type(wires)!=list):
            raise ValueError('type of wires should be list')
        if(len(wires)!=circuit.num_wires):
            raise ValueError('len(wires) should be equal to circuit.num_wires')
        if(len(wires)>self.num_wires):
            raise ValueError('len(wires) should be no larger than self.num_wires')
        if(not(np.max(wires)<self.num_wires and np.min(wires)>=0)):
            raise ValueError('some wire numbers in wires are out of range')
        for op in circuit.op_queue:
            new_wires=[wires[x] for x in op.wires]
            op.wires=new_wires
            self.op_queue.append(op)

    """def extend(self, operation, wires):
        if(type(wires)!=list):
            raise ValueError('type of wires should be list')
        if(len(wires)!=circuit.num_wires):
            raise ValueError('len(wires) should be equal to circuit.num_wires')
        if(len(wires)>self.num_wires):
            raise ValueError('len(wires) should be no larger than self.num_wires')
        if(not(np.max(wires)<self.num_wires and np.min(wires)>=0)):
            raise ValueError('some wire numbers in wires are out of range')
        new_wires=[wires[x] for x in operation.wires]
        operation.wires=new_wires
        self.op_queue.append(op)
        """
    def calcUnitary(self):
        U = 1
        for i in range(self.num_wires):
            U = np.kron(U, IdentityGate)
        for op in self.op_queue:
            #op is a class, num_wires, wires, op
            for wire in op.wires:
                if(wire<0 or wire>=self.num_wires):
                    raise ValueError('Wire %s out of range' %(str(op.wires)))
            if(op.num_wires==1):
                tmp=1+0j
                if(len(op.op)>1):
                    for i in range(self.num_wires):
                        tmp=np.kron(tmp, IdentityGate)
                    for o in op.op:
                        tmp0=1+0j
                        for i in range(self.num_wires):
                            if(op.wires[0]==i):
                                tmp0=np.kron(tmp0, o[0])
                            else:
                                tmp0=np.kron(tmp0, IdentityGate)
                        tmp=np.dot(tmp0, tmp)
                else:
                    for i in range(self.num_wires):
                        if(op.wires[0]==i):
                            tmp=np.kron(tmp, op.op[0])
                        else:
                            tmp=np.kron(tmp, IdentityGate)
                U=np.dot(tmp, U)
            else:
                tmp=np.zeros(U.shape)*0j
                ind=[]
                for i in range(self.num_wires):
                    if (i in op.wires):
                        ind.append(op.wires.index(i))
                    else:
                        ind.append(-1)
                for op0 in op.op:#op0:list, a tensor product
                    tmp1=1
                    for i in range(self.num_wires):
                        if(ind[i]==-1):
                            tmp1=np.kron(tmp1, IdentityGate)
                        else:
                            tmp1=np.kron(tmp1, op0[ind[i]])
                    #print(tmp, tmp1)
                    tmp+=tmp1
                U=np.dot(tmp, U)
        return U

    def execute(self, measure_wires=None, shot=None, style='probability'):
        u=self.calcUnitary()
        if(self.initial_state is None):
            raise ValueError('initial_state not exist')
        finalDensityMatrix=np.dot(np.dot(u,self.initial_state.densityMatrix), u.conjugate().T)
        if(measure_wires is None):
            raise ValueError('Operator undefined')
        if(type(measure_wires) is int):
            measure_wires=[measure_wires]
        measure=1
        for i in range(self.num_wires):
            if(i in measure_wires):
                measure=np.kron(measure, proj00)
            else:
                measure=np.kron(measure, IdentityGate)
        res=np.trace(np.dot(measure,finalDensityMatrix))
        res=np.sum(np.sqrt(res.imag**2+res.real**2))
        if(shot is None):
            if (style == 'probability'):
                return res
            elif (style == 'expectation'):
                return 2*res-1
        else:
            if(style=='probability'):
                return np.sum(np.array([np.random.rand()<res for x in range(shot)]))/shot
            elif(style=='expectation'):#return expectation value, need fixing later because actually it's Hadamard+\sigma_z instead of \sigma_x
                return np.sum(np.array([(-1+(2)*(np.random.rand()<res)) for x in range(shot)]))/shot

    def expectation(self, H):
        u=self.calcUnitary()
        if(self.initial_state is None):
            raise ValueError('initial_state not exist')
        finalState=np.dot(u, self.initial_state.state)
        expect=np.dot(finalState.conj().T, np.dot(H, finalState))
        return expect
        
    
class CSWAP(Circuit):
    def __init__(self, initial_state=None):
        self.num_wires=3
        self.op_queue=[]
        if(initial_state is None):
            self.initial_state=None
        elif(type(initial_state)==State):
            self.initial_state=initial_state
        else:
            raise ValueError('initial state type error')
        self.add(H(0))
        self.add(CNOT([1, 2]))
        V = np.exp(-1j * np.pi / 4) * (IdentityGate + 1j * NotGate) / np.sqrt(2)
        self.add(CtrlUnitary([2, 1], V))
        self.add(CNOT([0, 2]))
        self.add(CtrlUnitary([2, 1], V.conjugate().T))
        self.add(CNOT([0, 2]))
        self.add(CtrlUnitary([0, 1], V))
        self.add(CNOT([1, 2]))
        self.add(H(0))

    def fidelity(self,shot=None):
        return np.sqrt(self.execute([0],shot=shot)*2-1)

class ArbitraryOneQubitGate(Circuit):
    def __init__(self, params=None):
        self.num_wires=1
        self.op_queue=[]
        self.initial_state=None
        if(params is None):
            self.add(Rz((np.random.rand()-0.5)*np.pi, wires=[0]))
            self.add(Ry((np.random.rand() - 0.5) * np.pi, wires=[0]))
            #self.add(Rz((np.random.rand() - 0.5) * np.pi, wires=[0]))
            self.add(Rx((np.random.rand() - 0.5) * np.pi, wires=[0]))
        else:
            self.add(Rz(params[0], wires=[0]))
            self.add(Ry(params[1], wires=[0]))
            #self.add(Rz(params[2], wires=[0]))
            self.add(Rx(params[2], wires=[0]))


"""general two qubit gate: arXiv 0308006 FIG. 7"""
class ArbitraryTwoQubitGate(Circuit):
    def __init__(self, params=None):
        self.num_wires=2
        self.op_queue=[]
        self.initial_state=None
        if(params is None):
            # A1, A2
            self.add([Rz(param=(np.random.rand() - 0.5) * np.pi,wires=[0]), Rz(param=(np.random.rand() - 0.5) * np.pi,wires=[1])])
            self.add([Ry(param=(np.random.rand() - 0.5) * np.pi,wires=[0]), Ry(param=(np.random.rand() - 0.5) * np.pi,wires=[1])])
            self.add([Rz(param=(np.random.rand() - 0.5) * np.pi,wires=[0]), Rz(param=(np.random.rand() - 0.5) * np.pi,wires=[1])])
            self.add(CNOT([1,0]))
            self.add([Rz(param=(np.random.rand() - 0.5) * np.pi,wires=[0]), Ry(param=(np.random.rand() - 0.5) * np.pi,wires=[1])])
            self.add(CNOT([0,1]))
            self.add(Ry(param=(np.random.rand() - 0.5) * np.pi,wires=[1]))
            self.add(CNOT([1,0]))
            #A3, A4
            self.add([Rz(param=(np.random.rand() - 0.5) * np.pi,wires=[0]), Rz(param=(np.random.rand() - 0.5) * np.pi,wires=[1])])
            self.add([Ry(param=(np.random.rand() - 0.5) * np.pi,wires=[0]), Ry(param=(np.random.rand() - 0.5) * np.pi,wires=[1])])
            self.add([Rz(param=(np.random.rand() - 0.5) * np.pi,wires=[0]), Rz(param=(np.random.rand() - 0.5) * np.pi,wires=[1])])
        else:
            self.add([Rz(param=params[0],wires=[0]), Rz(param=params[1],wires=[1])])
            self.add([Ry(param=params[2],wires=[0]), Ry(param=params[3],wires=[1])])
            self.add([Rz(param=params[4],wires=[0]), Rz(param=params[5],wires=[1])])
            self.add(CNOT([1, 0]))
            self.add([Rz(param=params[6],wires=[0]), Ry(param=params[7],wires=[1])])
            self.add(CNOT([0, 1]))
            self.add(Ry(param=params[8],wires=[1]))
            self.add(CNOT([1, 0]))
            # A3, A4
            self.add([Rz(param=params[9],wires=[0]), Rz(param=params[10],wires=[1])])
            self.add([Ry(param=params[11],wires=[0]), Ry(param=params[12],wires=[1])])
            self.add([Rz(param=params[13],wires=[0]), Rz(param=params[14],wires=[1])])

class BasicBlockTwoQubitGate(Circuit):
    """
    a block: a entangled gate (CNOT) followed by one single qubit gate on each wire
    num_blocks: number of blocks
        number of total parameters=num_blocks*2
        depth=num_blocks*2

    """
    def __init__(self, num_blocks=1, params=None):
        self.num_wires=2
        self.op_queue=[]
        self.initial_state=None
        if(params is None):
            for i in range(num_blocks):
                self.add(CNOT([0,1]))
                self.add(Rz(param=(np.random.rand() - 0.5) * np.pi, wires=[0]))
                self.add(Ry(param=(np.random.rand() - 0.5) * np.pi, wires=[0]))
                self.add(Rz(param=(np.random.rand() - 0.5) * np.pi, wires=[0]))
                self.add(Rz(param=(np.random.rand() - 0.5) * np.pi, wires=[1]))
                self.add(Ry(param=(np.random.rand() - 0.5) * np.pi, wires=[1]))
                self.add(Rz(param=(np.random.rand() - 0.5) * np.pi, wires=[1]))
        else:
            for i in range(num_blocks):
                self.add(CNOT([0,1]))
                self.add(Rz(param=params[i*6+0], wires=[0]))
                self.add(Ry(param=params[i*6+1], wires=[0]))
                self.add(Rz(param=params[i*6+2], wires=[0]))
                self.add(Rz(param=params[i*6+3], wires=[1]))
                self.add(Ry(param=params[i*6+4], wires=[1]))
                self.add(Rz(param=params[i*6+5], wires=[1]))


