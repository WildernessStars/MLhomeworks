import os
k=5
N=5

avgrank=[3.2, 3.8, 1.2, 4, 2.8]
def friedman(k, N):
    critical=3.007
    temp=0
    K=(k+1)/2
    for i in range(0,k-1):
        temp+=pow(avgrank[i]-K,2)
    temp=((k-1)*12*N)/(k*k*k-k)*temp
    temp=((N-1)*temp)/(N*(k-1)-temp)
    print('friedman:',temp)
    if temp>critical:
        return False
    return True

def nemenyi(k, N):
    q=2.728
    cd=((k*(k+1)/N)/6)**0.5
    cd=q*cd
    print('cd=',cd)
    if cd<maxgap():
        return False
    return True


def maxgap():
    maxr=0
    minr=1000
    for i in range(0,k-1):
        if avgrank[i]>maxr:
            maxr=avgrank[i]
        if avgrank[i]<minr:
            minr=avgrank[i]
    print('maxgap=',maxr-minr)
    return maxr-minr

if __name__ == '__main__':
    if friedman(k, N)==True:
        print('true')
    else:
        if nemenyi(k, N)==True:
            print('true')
        else:
            print('false')
    os.system("pause")