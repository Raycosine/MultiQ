import numpy as np
###NORMALIZATION!!!
from abc import ABC, ABCMeta, abstractmethod, abstractproperty
"""one qubit gate/operation"""
HadamardGate=1/np.sqrt(2)*np.array([[1,1],[1,-1]])
IdentityGate=np.array([[1,0],[0,1]])
NotGate=np.array([[0,1],[1,0]])#also pauli-x
zero=np.array([1,0]).reshape(2,1)
"""projection operator + |0><1|, |1><0|"""
proj00=np.array([[1,0],[0,0]])#|0><0|
proj11=np.array([[0,0],[0,1]])
proj01=np.array([[0,1],[0,0]])
proj10=np.array([[0,0],[1,0]])
"""for convenience, some multiple-qubit gate, only used in old versions of code"""
HII=np.kron(np.kron(HadamardGate,IdentityGate),IdentityGate)
NII=np.kron(np.kron(NotGate,IdentityGate),IdentityGate)
E0II=np.kron(np.array([[1,0],[0,0]]),np.kron(IdentityGate, IdentityGate))
FredkinGate=np.array(
               [[1,0,0,0,0,0,0,0],
                [0,1,0,0,0,0,0,0],
                [0,0,1,0,0,0,0,0],
                [0,0,0,1,0,0,0,0],
                [0,0,0,0,1,0,0,0],
                [0,0,0,0,0,0,1,0],
                [0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,1]])
switch_swap={
    0:(proj00,proj00),
    1:(proj01,proj10),
    2:(proj11,proj11),
    3:(proj10,proj01)
}


"""
    operation classes
    self: wires, num_wires, op
    wires: list of int, wires the operator applied on
    num_wires: int, number of the wires
    op: list(num_op*num_wires) of np.ndarry
        1 or multi qubit, decomposed into a sum of tensor product   
    num_wires should be defined once the dimension of operator is certain
"""
class Operation(ABC):
    """
    __init__, defined inside
    """
    """
    __init__, operator defined, wires not sure
    """
    def __init__(self, wires=None):
        if (wires is not None):
            if (type(wires) == int):#single qubit, single wire
                if (self.num_wires != 1):#dim of operator != num of wires
                    raise ValueError('1 wires used for %d-qubit gate' % self.num_wires)
                self.wires = [wires]
            else:
                if(type(wires)!=list):
                    raise ValueError('Parameter wires should be list or int')
                else:#single or multi qubit, wires is a list
                    if (len(wires) != self.num_wires):
                        raise ValueError('>%d wires used for %d-qubit gate' % (len(wires), self.num_wires))
                    self.wires = wires
        else:
            raise ValueError('Parameter ''wires'' not defined, %d wires required' % self.num_wires)
    """
    __init__ operator is defined by parameters, wires not sure
    example: Rx, Ry, Rz
    example: Controlled unitary
    different kinds of operators use parameters differently
    """
    def __init__(self, wires=None, param=None):
        pass
    """
    __init__ self-defined operator, wires not sure
    """
    def __init__(self, wires=None, op=None):
        pass

    def set_wires(self, wires=None):
        if (wires is None):
            raise ValueError('Wires not defined')
        else:
            if (type(wires) == int):
                if (self.num_wires != 1):
                    raise ValueError('>1 wires used for 1 qubit operation')
                self.wires = [wires]
            elif (type(wires) == list):
                if (self.num_wires != len(wires)):
                    raise ValueError('%d wires used for %d-qubit operation' % (len(wires), self.num_wires))
                self.wires = wires
            else:
                raise ValueError('wires type should be int or list of int')
    """
    set_op:
        num_wires is already determined
    """
    def set_op(self, op):
        if (op is None):
            raise ValueError('Operation not defined')
        else:
            if (type(op) == np.ndarray):
                if (self.num_wires != 1):
                    raise ValueError('1-qubit operator used for multi-qubit gate')
                if (len(op.shape) != 2 or op.shape[1] != 2 ** self.num_wires):
                    raise ValueError(
                        'give operator has shape %s while number of wires is %s' % (str(op.shape), str(self.num_wires)))
                self.op = [op]
            elif (type(op) == list):  # sum of tensor product
                if (type(op[0]) == np.ndarray):  # only one element(tensor product)
                    if (len(op) != self.num_wires):
                        raise ValueError('Operator dim != num_wires')
                    for op0 in op:
                        if (type(op0) != np.ndarray):
                            raise ValueError('wrong operator type')
                        if (op0.shape != (2, 2)):
                            raise ValueError('tensor should be 2*2')
                    self.op = [op]
                elif (type(op[0]) == list):
                    for op0 in op:
                        if (len(op0) != self.num_wires):
                            raise ValueError('Operator dim != num_wires')
                        for op00 in op0:
                            if (type(op00) != np.ndarray):
                                raise ValueError('wrong operator type')
                            if (op00.shape != (2, 2)):
                                raise ValueError('tensor should be 2*2')
                    self.op = [op]
                else:
                    raise ValueError('wrong operator type')
            else:
                raise ValueError('wrong operator type')


class I(Operation):
    num_wires = 1
    op = [np.array([[1, 0], [0, 1]])]

    def __init__(self, wires=None):
        self.set_wires(wires)


class H(Operation):
    num_wires = 1
    op = [np.array([[1, 1], [1, -1]]) / np.sqrt(2)]

    def __init__(self, wires=None):
        self.set_wires(wires)

class Z(Operation):
    num_wires = 1
    op = [np.array([[1, 0], [0, -1]])]

    def __init__(self, wires=None):
        self.set_wires(wires)


class S(Operation):
    num_wires = 1
    op = [np.array([[1, 0], [0, 1j]])]

    def __init__(self, wires=None):
        self.set_wires(wires)


class CNOT(Operation):
    num_wires = 2
    op = [[proj00,
           IdentityGate],
          [proj11,
           NotGate]]
    """op=np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,0,1],
                 [0,0,1,0]])"""

    def __init__(self, wires=None):
        self.set_wires(wires)



class CZ(Operation):
    num_wires = 2
    op = [[proj00,
           IdentityGate],
          [proj11,
           np.array([[1, 0], [0, -1]])]]
    """op=np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,0,1],
                 [0,0,1,0]])"""

    def __init__(self, wires=None):
        self.set_wires(wires)

class CtrlUnitary(Operation):
    num_wires = 2
    def __init__(self, wires=None, param=None):#param must be 2*2 unitary
        self.set_wires(wires)
        """Now we should deal with operation"""
        if(param is None):
            raise ValueError('Operation not defined')
        else:
            if(type(param)!=np.ndarray):
                raise ValueError('wrong operation type')
            if(len(param.shape)!=2 or (len(param.shape)==2 and param.shape!=(2,2))):
                raise ValueError('U of control-U should be 2*2')
            self.op = [[proj00,IdentityGate],[proj11,param]]

class Rx(Operation):
    num_wires = 1

    def __init__(self, param=None, wires=None):
        self.set_wires(wires)
        if (param is not None and np.isreal(param)):
            self.op = [np.array([[np.cos(param / 2), -1j * np.sin(param / 2)],
                                [-1j * np.sin(param / 2), np.cos(param / 2)]])]
        else:
            raise ValueError('parameter undefined or type error')


class Ry(Operation):
    num_wires = 1

    def __init__(self, param=None, wires=None):
        self.set_wires(wires)
        if (param is not None and np.isreal(param)):
            self.op = [np.array([[np.cos(param / 2), -np.sin(param / 2)],
                                [np.sin(param / 2), np.cos(param / 2)]])]
        else:
            raise ValueError('parameter undefined or type error')


class Rz(Operation):
    num_wires = 1

    def __init__(self, param=None, wires=None, op=None):
        self.set_wires(wires)
        if (param is not None and np.isreal(param)):
            self.op = [np.array([[np.exp(-1j * param / 2), 0],
                                [0, np.exp(1j * param / 2)]])]
        else:
            raise ValueError('parameter undefined or type error')


class U3(Operation):
    num_wires=1
    
    def __init__(self, param=None, wires=None, op=None):
        self.set_wires(wires)
        if (param is not None and np.sum(np.isreal(param))==len(param)):
            self.op = [Rz(param[0],wires).op, Ry(param[1],wires).op, Rz(param[2],wires).op]
        else:
            raise ValueError('parameter undefined or type error')


class SWAP(Operation):

    def __init__(self, ctrl=None, tgt1=None, tgt2=None):#ctrl: control qubit, 1 wire; tgt1&tgt2: target qubits, n(>=1) wires
        """
        :int ctrl: wire num of the control qubit
        :int or list of int tgt1:wire num of qubits of the first target
        :int or list of int tgt2:wire num of qubits of the second target
        len(tgt1)==len(tgt2)
        """
        if(type(ctrl)!=int):
            raise ValueError('wire number of the control qubit(ctrl) should be int')
        if(type(tgt1)==int):
            tgt1=[tgt1]
        elif(type(tgt1)!=list):
            raise ValueError('wire(s) of the target qubits(tgt1) should be int or list')
        if(type(tgt2)==int):
            tgt2=[tgt2]
        elif(type(tgt2)!=list):
            raise ValueError('wire(s) of the target qubits(tgt2) should be int or list')
        if(len(tgt1)!=len(tgt2)):
            raise ValueError('numbers of wire(s) of the target qubits don''t match')
        self.num_wires=1+len(tgt1)+len(tgt2)
        tmpwires=[0]
        tmpwires.extend(tgt1)
        tmpwires.extend(tgt2)
        self.wires=tmpwires
        op0=[proj00]
        for i in range(len(tgt1)+len(tgt2)):
            op0.append(IdentityGate)
        self.op=[op0]

        for tmpcount in range(np.power(4,len(tgt1))):
            op1=[proj11]
            opa=[]
            opb=[]
            tmpcount0=tmpcount
            for tmp in range(len(tgt1)):
                tmpa,tmpb=switch_swap[tmpcount0%4]
                opa.append(tmpa)
                opb.append(tmpb)
                tmpcount0=tmpcount0//4
            op1.extend(opa)
            op1.extend(opb)
            self.op.append(op1)
