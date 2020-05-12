import pickle
import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
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
    cnt=0

    plt.figure()
    for M,N in MN:
        cnt+=1
        #if(cnt>=3):
        #    continue
        F=[]
        print('M=%d' % M, 'N=%d' % N)
        for i in range(7):
            if(i+1!=7):
                continue
            filename='singleRoundResult\\%d\\M=%d_N=%d.pickle' % (i+1,M,N)
            d=pickle.load(open(filename, 'rb'))
            F.extend(d['F'])
        F=np.array(F)
        if(N==1):
            F=np.sqrt(F*2-1)
        G=d['G']
        G=np.array(G)
        averF=np.average(F)
        #F=F-averF
        palette = plt.get_cmap('hsv')
        plt.plot([averF,averF], [0,60],'-',alpha=0.5, color=palette(cnt*30))
        n, bins, pathces = plt.hist(F, 50, density=True, alpha=0.5, color=palette(cnt*30), label='M=%d N=%d' % (M, N))
        print(len(F), len(G), len(G[0]))
    plt.legend()
    plt.show()