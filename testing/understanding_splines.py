import numpy as np

t_ref=[0,1,2,3]
y_ref=[4,2,3,6]

k_max = 3
n=len(t_ref)


t_eval = 1.5

B = np.zeros((k_max,n))

for k in range(0,k_max):
    for i in range(0,n):
        if(k==0):
            if(t_eval >t_ref[i] and t_eval < t_ref[i+1]):
                B[k,i]=1
        else:
            temp1 = (t_eval-t_ref[i]/(t_ref[i+k] -t_ref[i])) * B[k-1,i]
            temp2 = ((t_ref[i+k+1]-t_eval)/(t_ref[i+k+1] -t_ref[i+1]))  * B[k-1,i+1]
            B[k,i] =  temp1 + temp2


print(B)
