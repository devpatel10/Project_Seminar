#####################################################################
#               SOURCE CODE                                         #
#           Patel Devarshi Chandrakant  | 18CS10040                 #
#           Project Seminar                                         #
#           Machine Learning based modeling and                     #
#           interpretation of data from wearable sensor             #
#                                                                   #
#          Mentored by : Mrs. Saswati Pal                           #
#          Supervised by : Prof. Sudip Misra                        #
#                                                                   #
#####################################################################
import scipy.io
import numpy as np 
from matplotlib import pyplot as plt
import os.path

# Dataset used https://www.kaggle.com/bjoernjostein/china-12lead-ecg-challenge-database
# China 12-Lead ECG Challenge Database

# Loops through  entire dataset 
# To see a specific training instance remove the outer most loop 
# It is customized with respect to a particular dataset
# Test files are assumed to be in a folder Training_2 in the same directory
# And the files are named Q0001.hea--->Q3581.hea

# Writes the result into a file values.txt

f = open("values.txt", "w")
titles = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
for i in range(3581):
    k="000"+str(i+1)+".mat"
    s="Training_2/Q"+k[-8:]
    if(os.path.isfile(s)):
        mat = scipy.io.loadmat(s)
        y=np.empty(shape=(13,mat["val"][0].size))
        plot1=plt.figure(1)
        y[0]=np.ones(shape=(1,mat["val"][0].size))
        x=np.arange(1,1+mat["val"][0].size)
        for i in range(12):
            y[i+1]=mat["val"][i]/1000
            plt.subplot(3, 4, 1+i)
            plt.title(titles[i])
            plt.plot(x,y[i+1]) 
        plt.show()

        #Printing aVR, aVL, aVF predicted
        plot2=plt.figure(2)
        plt.subplot(1,3,1)
        y1=(-1*(y[1]+y[2])/2)
        print("Absolute avg aVR squared error : ",np.sum(np.square(y1))/mat["val"][0].size)
        plt.title("aVR Predicted")
        plt.plot(x,y1) 
        plt.subplot(1,3,2)
        plt.title("aVL Predicted")
        y2=((y[1]-y[3])/2)
        print("Absolute avg aVL squared error : ",np.sum(np.square(y2))/mat["val"][0].size)
        plt.plot(x,y2) 
        plt.subplot(1,3,3)
        plt.title("aVF Predicted")
        y3=((y[2]+y[3])/2)
        print("Absolute avg aVF squared error : ",np.sum(np.square(y3))/mat["val"][0].size)
        plt.plot(x,y3)
        plt.show()


        #Priting hypothesis parameter for each unipolar leads V1 to V6
        X=y[[1,2,3]]
        X=X.T
        d1=np.diff(X,axis=0)
        d1 = np.insert(d1, 0, 0., axis=0)
        d1=d1*10
        X=np.concatenate((X,d1),axis=1)  #appending d(I)/dt,d(II)/dt,d(III)/dt


        #  For appending d2(I)/dt2,d2(II)/dt2,d2(III)/dt2
        # d2=np.diff(d1,axis=0)
        # d2 = np.insert(d2, 0, 0., axis=0)
        # d2=d2/10
        # X=np.concatenate((X,d2),axis=1)

        # Appending column to incorpoarate constant parameter
        X = np.insert(X, 0, 1., axis=1)
        for t in range(7,12) :
            Y=y[t].reshape(mat["val"][0].size,1)

            # Preprocessing 
            # making sure X is not singleton
            X_refined=np.unique(X,axis=0)
            Y_refined=np.zeros(shape=(X_refined.shape[0],1))
            temp_array=np.zeros(shape=(X_refined.shape[0],1))
            #   print(X_refined.shape,X_refined)
            # If the value of instance is same then taking the avg of all such output values to
            for itr in range(X.shape[0]) :
                j=np.where(np.all(X_refined==X[itr],axis=1))
                temp_array[j[0]]=temp_array[j[0]]+1
                Y_refined[j[0]]=np.add(Y_refined[j[0]],Y[itr]) 
            Y_refined=np.divide(Y_refined,temp_array)  
            # print(X_refined,Y_refined)
            # print(max(Y_refined),min(Y_refined))
            # print(Y_refined.shape,np.linalg.det(np.dot(X_refined.T,X_refined)))

            # Normal Equation of Linear Regression
            theta=np.dot(np.dot(np.linalg.inv(np.dot(X_refined.T,X_refined)),X_refined.T),Y_refined)
            Y_estimated=np.dot(X,theta)
            # print(Y_estimated)
            plot2=plt.figure(3)
            plt.plot(x,y[t],label='Actual')
            plt.title("V"+str(t-6))
            plt.plot(x,Y_estimated,label='Prdicted') 
            plt.legend()
            # print(mat["val"][0].size)
            print("Squared Error : ", np.sum(np.square(Y_estimated-y[7]))/mat["val"][0].size)
            plt.show()
            f.write(np.array_str(theta))
            f.write("\n")
    else : 
        print("No file ",s," Found")
f.close()