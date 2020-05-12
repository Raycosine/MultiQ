from Circuit import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
"""
Single Round Analysis
Compare M*N & N_tot 
20*50... 1000
for each setting, we run 1000 times for single cost function, 
    and for single gradient
        over different estimated channel and given unknown channel
        (however we fix one channel due to the symmetry(?))

"""

def SingleRun(theta0, theta1, shot=None):

    random1 = randomState(dim=1)
    C1 = Circuit(num_wires=3)
    # U: unknown channel
    C1.extend(ArbitraryOneQubitGate(params=theta0), wires=[1])
    # \tilde U: estimated channel
    C1.extend(ArbitraryOneQubitGate(params=theta1), wires=[2])

    # C-SWAP test
    C1.add(H(0))
    C1.add(SWAP(ctrl=0, tgt1=[1], tgt2=[2]))
    C1.add(H(0))
    # prepare initial state as two same input
    C1.setInitialState(State(dim=3, state=np.kron(np.kron(np.array([1, 0]).reshape(2, 1), random1), random1)))
    #return C1.execute(measure_wires=[0], shot=shot)
    ans1=C1.execute(measure_wires=[0], shot=shot)

    C2 = Circuit(num_wires=3)
    # U: unknown channel
    #C2.extend(ArbitraryOneQubitGate(params=theta0), wires=[1])
    # \tilde U: estimated channel
    #C2.extend(ArbitraryOneQubitGate(params=theta1), wires=[2])

    # C-SWAP test
    C2.add(H(0))
    C2.add(SWAP(ctrl=0, tgt1=[1], tgt2=[2]))
    C2.add(H(0))
    # prepare initial state as two same input
    UnknownChannel=ArbitraryOneQubitGate(params=theta0)
    TrueOutput=np.dot(UnknownChannel.calcUnitary(), random1)
    EstimatedChannel=ArbitraryOneQubitGate(params=theta1)
    EstimatedOutput=np.dot(EstimatedChannel.calcUnitary(), random1)

    C2.setInitialState(State(dim=3, state=np.kron(np.kron(np.array([1, 0]).reshape(2, 1), TrueOutput), EstimatedOutput)))
    # return C1.execute(measure_wires=[0], shot=shot)
    ans2 = C2.execute(measure_wires=[0], shot=shot)
    #print(ans1, ans2)
    return ans1

if __name__ == '__main__':
    MN=[
        (1000,1),
        (500,2),
        (200,5),
        (100,10),
        (50,20),
        (20,50),
        (10,100),
        (5,200),
        (2,500),
        (1,1000)
    ]
    alpha=0.1
    #iteration
    ICF=[0]*50
    TCF=[0]*50
    GRADIENT=[0]*50
    THETA=[0]*50
    dct={}
    #fixed channels for every single run
    theta0 = [-1.5,0.5,-1.5]#(np.random.rand(3) - 0.5) * np.pi
    theta1 = [1,1,1]
    for M,N in MN:
        print(M,N)
        F=[]
        G=[[],[],[]]
        for test in range(400):
            #if you want to use different channels for each single test, delete the #s below
            #theta0 = (np.random.rand(3) - 0.5) * np.pi
            # theta0=np.array([0,0,0])
            # theta1=(np.random.rand(3) - 0.5) * np.pi
            #theta1 = np.array([1, 1, 1])#theta1 fixed
            tf=0.0
            for i in range(M):
                singlef=SingleRun(theta0, theta1, shot=N)
                if(singlef>=0.5):
                    singlef=np.sqrt(singlef*2-1)
                else:
                    singlef=0
                tf += singlef
            tf=tf/M
            F.append(tf)

            if(test%50==0):
                print('Test', test, 'Fidelity', tf)
            for thetai in range(3):
                theta_plus= [theta1[j]+(j==thetai)*np.pi/2 for j in range(3)]
                theta_minus=[theta1[j]-(j==thetai)*np.pi/2 for j in range(3)]
                gplus=0
                for i in range(M):
                    singlef=SingleRun(theta0, theta_plus, shot=N)
                    if(singlef>=0.5):
                        singlef=np.sqrt(singlef*2-1)
                    else:
                        singlef=0
                    gplus+=singlef
                gplus/=M

                gminus = 0
                for i in range(M):
                    singlef = SingleRun(theta0, theta_minus, shot=N)
                    if (singlef >= 0.5):
                        singlef = np.sqrt(singlef * 2 - 1)
                    else:
                        singlef = 0
                    gminus += singlef
                gminus /= M
                G[thetai].append(1/np.sqrt(2)*(gplus-gminus))

        d={}
        d['F']=F
        d['G']=G
        pickle.dump(d, open("M=%d_N=%d.pickle" % (M, N), "wb"))

