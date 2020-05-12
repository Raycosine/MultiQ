from Circuit import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
"""
Two Qubit Channel, Single shot
"""

def SingleRun(theta0, theta1, num_blocks, shot=None,style='expectation'):
    C2 = Circuit(num_wires=5)
    C2.extend(ArbitraryTwoQubitGate(params=theta0), wires=[1, 2])
    C2.extend(BasicBlockTwoQubitGate(num_blocks=num_blocks, params=theta1), wires=[3, 4])
    C2.add(H(0))
    C2.add(SWAP(ctrl=0, tgt1=[1, 2], tgt2=[3, 4]))
    C2.add(H(0))
    random2 = randomState(dim=2)
    C2.setInitialState(State(dim=5, state=np.kron(np.kron(np.array([1, 0]).reshape(2, 1), random2), random2)))
    return C2.execute(measure_wires=[0], shot=shot,style=style)

if __name__ == '__main__':
    #initial
    ROUND=100
    REALM=100
    MC=20
    MG=20
    N=1
    alpha=0.1
    num_blocks=7
    num_params=num_blocks*6
    #iteration
    ICF=[0]*ROUND
    TCF=[0]*ROUND
    GRADIENT=[0]*ROUND
    #THETA=[0]*50
    THETA=[]
    CNTM=[]
    plt.figure()
    idealCF=[]
    realCF=[]
    theta0 = np.array([-1.5 + 2 * (x % 2 == 1) for x in range(15)])
    theta1 = np.zeros(num_params)+1
    for test in range(100):
        print('Test',test)
        Round=0
        singlei=[]
        singlet=[]
        singleM=[]
        totM=0
        alpha=1
        while(1):
            if(Round==20):#we set the maximum num of pairs, so we don't need round limit
                break
            #ideal cost function
            icf=0.0
            for i in range(REALM):
                icf+=SingleRun(theta0, theta1,num_blocks, style='expectation')
            icf/=REALM
            icf=1-icf
            singlei.append(icf)
            ICF[Round]+=icf
            #real cost function, MC training pairs are used as the test dataset at each round
            tcf=0.0
            for i in range(MC):
                tcf+=SingleRun(theta0, theta1, num_blocks,shot=N,style='expectation')
            tcf/=MC
            tcf=1-tcf
            singlet.append(tcf)
            TCF[Round]+=tcf
            if(Round>0):
                if(singlet[-1]<singlet[-2]):
                    alpha*=1.05
                else:
                    alpha*=0.95
            singleM.append(totM)
            totM+=MC
            gradient=[]
            for i in range(num_params):
                theta_plus=[theta1[j]+(j==i)*np.pi/2 for j in range(num_params)]
                theta_minus = [theta1[j]-(j==i)*np.pi/2 for j in range(num_params)]
                gplus=0
                for i in range(MG):
                    gplus+=SingleRun(theta0, theta_plus, num_blocks,shot=N,style='expectation')
                gplus/=MG
                gminus=0
                for i in range(MG):
                    gminus+=SingleRun(theta0, theta_minus, num_blocks,shot=N,style='expectation')
                gminus/=MG
                gradient.append(np.sqrt(2)*(gplus-gminus))
                totM+=2*MG
            theta1=theta1+np.array(gradient)*alpha
            Round+=1
            #print('Gradient:', gradient)
            if(Round%2==0):
                print('Cost function',icf, tcf)
                print('Theta:', theta1)
            #GRADIENT[Round]+=gradient
            #THETA.append(theta1)
        THETA.append(theta1)
        CNTM.append(singleM)
        idealCF.append(singlei)
        realCF.append(singlet)
        print(test,singlei[-1], singlet[-1],theta1)
        #aver=np.abs(np.average(THETA, axis=0)-np.array(theta0))
    pickle.dump((THETA, CNTM,idealCF, realCF), open("quantum_result0\\TWOQ_blocks%d_1.pickle" % num_blocks, "wb"))
