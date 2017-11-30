
import numpy as np
import time
import matplotlib.pyplot as plt

# first function to fill, compute distance matrix using loop
def compute_distance_naive(X):
    N = X.shape[0]      # num of rows
    D = X[0].shape[0]   # num of cols

    M = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            xi = X[i,:]
            xj = X[j,:]
            M[i,j] = np.sqrt(np.dot((xi-xj).T,(xi-xj)))

    return M



# second function to fill, compute distance matrix without loops
def compute_distance_smart(X):
    N = X.shape[0]  # num of rows
    D = X[0].shape[0]  # num of cols
    
    # use X to create M
    XX = (X * X).sum(axis=1)[:, np.newaxis]
    YY = XX.T
    Z = -2*np.dot(X, X.T) + XX + YY
    np.maximum(Z, 0, out=Z) 
    Z.flat[::Z.shape[0] + 1] = 0.0 
    Z = np.sqrt(Z)
    return Z

# third function to fill, compute correlation matrix using loops
def compute_correlation_naive(X):
    N = X.shape[0]  # num of rows
    D = X[0].shape[0]  # num of cols

    M = np.zeros([D, D])
    for i in range(D):
        for j in range(D):
            xi = X[:, i]
            xj = X[:, j]
            mui = np.sum(xi).astype(float)/ N
            muj = np.sum(xj).astype(float)/N
            xni= xi- mui
            xnj = xj- muj
            zij = (np.dot(xni, xnj)).astype(float)/ (N-1)
            sigi = np.sqrt(np.dot(xni, xni).astype(float)/(N-1))
            sigj = np.sqrt(np.dot(xnj, xnj).astype(float)/(N-1))
            sig = sigi * sigj
            Zmatrix = zij.astype(float)/sig
            M[i,j] = Zmatrix
                           
            
    return M


# fourth function to fill, compute correlation matrix without loops
def compute_correlation_smart(X):
    N = X.shape[0]  # num of rows
    D = X[0].shape[0]  # num of cols
    M = np.zeros([D,D])
    vectormu = (np.sum(X, axis = 0)).astype(float)/N
    matrixmu = vectormu * np.ones([N,1])
    x = X- matrixmu
    xT = np.transpose(x)
    cov= (np.dot(xT,x)).astype(float) / (N-1)
    x2 = np.multiply(x,x)
    varience = (np.sum(x2, axis =0)).astype(float) / (N-1)
    sig = np.sqrt(varience)
    Zmatrix = np.outer(sig, sig)
    M = np.multiply(cov, np.power(Zmatrix, -1))
    return M



def main():
    print ('starting comparing distance computation .....')
    np.random.seed(100)
    params = range(10,141,10)   # different param setting
    nparams = len(params)       # number of different parameters

    perf_dist_loop = np.zeros([10,nparams])  # 10 trials = 10 rows, each parameter is a column
    perf_dist_cool = np.zeros([10,nparams])
    perf_corr_loop = np.zeros([10,nparams])  # 10 trials = 10 rows, each parameter is a column
    perf_corr_cool = np.zeros([10,nparams])

    counter = 0

    for ncols in params:
        nrows = ncols * 10

        print ("matrix dimensions: ", nrows, ncols)

        for i in range(10):
            X = np.random.rand(nrows, ncols)   # random matrix

            # compute distance matrices
            st = time.time()
            dist_loop = compute_distance_naive(X)
            et = time.time()
            perf_dist_loop[i,counter] = et - st              # time difference

            st = time.time()
            dist_cool = compute_distance_smart(X)
            et = time.time()
            perf_dist_cool[i,counter] = et - st

            assert np.allclose(dist_loop, dist_cool, atol=1e-06) # check if the two computed matrices are identical all the time

            # compute correlation matrices
            st = time.time()
            corr_loop = compute_correlation_naive(X)
            et = time.time()
            perf_corr_loop[i,counter] = et - st              # time difference

            st = time.time()
            corr_cool = compute_correlation_smart(X)
            et = time.time()
            perf_corr_cool[i,counter] = et - st
            assert np.allclose(corr_loop, corr_cool, atol=1e-06) # check if the two computed matrices are identical all the time

        counter = counter + 1

    mean_dist_loop = np.mean(perf_dist_loop, axis = 0)    # mean time for each parameter setting (over 10 trials)
    mean_dist_cool = np.mean(perf_dist_cool, axis = 0)
    std_dist_loop = np.std(perf_dist_loop, axis = 0)      # standard deviation
    std_dist_cool = np.std(perf_dist_cool, axis = 0)

    plt.figure(1)
    plt.errorbar(params, mean_dist_loop[0:nparams], yerr=std_dist_loop[0:nparams], color='red',label = 'Loop Solution for Distance Comp')
    plt.errorbar(params, mean_dist_cool[0:nparams], yerr=std_dist_cool[0:nparams], color='blue', label = 'Matrix Solution for Distance Comp')
    plt.xlabel('Number of Cols of the Matrix')
    plt.ylabel('Running Time (Seconds)')
    plt.title('Comparing Distance Computation Methods')
    plt.legend()
    plt.savefig('CompareDistanceCompFig.pdf')
 #   plt.show()    # uncomment this if you want to see it right way
    print ("result is written to CompareDistanceCompFig.pdf")

    mean_corr_loop = np.mean(perf_corr_loop, axis = 0)    # mean time for each parameter setting (over 10 trials)
    mean_corr_cool = np.mean(perf_corr_cool, axis = 0)
    std_corr_loop = np.std(perf_corr_loop, axis = 0)      # standard deviation
    std_corr_cool = np.std(perf_corr_cool, axis = 0)

    plt.figure(2)
    plt.errorbar(params, mean_corr_loop[0:nparams], yerr=std_corr_loop[0:nparams], color='red',label = 'Loop Solution for Correlation Comp')
    plt.errorbar(params, mean_corr_cool[0:nparams], yerr=std_corr_cool[0:nparams], color='blue', label = 'Matrix Solution for Correlation Comp')
    plt.xlabel('Number of Cols of the Matrix')
    plt.ylabel('Running Time (Seconds)')
    plt.title('Comparing Correlation Computation Methods')
    plt.legend()
    plt.savefig('CompareCorrelationCompFig.pdf')
 #   plt.show()    # uncomment this if you want to see it right way
    print ("result is written to CompareCorrelationCompFig.pdf")

import sklearn.datasets
iris = sklearn.datasets.load_iris()
print("Running iris pairwise loop...")
start_distance_loop_iris = time.time()
compute_distance_naive(iris.data)
total_time_distance_loop_iris = time.time() - start_distance_loop_iris

print("Running iris pairwise numpy...")
start_distance_numpy_iris = time.time()
compute_distance_smart(iris.data)
total_time_distance_numpy_iris = time.time() - start_distance_numpy_iris

print("Running iris correlation matrix loop...")
start_correlation_loop_iris = time.time()
compute_correlation_naive(iris.data)
total_time_correlation_loop_iris = time.time() - start_correlation_loop_iris


print("Running iris correlation matrix numpy...")
start_correlation_numpy_iris = time.time()
compute_correlation_smart(iris.data)
total_time_correlation_numpy_iris = time.time() - start_correlation_numpy_iris

breast_cancer = sklearn.datasets.load_breast_cancer()

print("Running breast cancer pairwise loop...")
start_distance_loop_breast_cancer = time.time()
compute_distance_naive(breast_cancer.data)
total_time_distance_loop_breast_cancer = time.time() - start_distance_loop_breast_cancer

print("Running breast cancer pairwise numpy...")
start_distance_numpy_breast_cancer = time.time()
compute_distance_smart(breast_cancer.data)
total_time_distance_numpy_breast_cancer = time.time() - start_distance_numpy_breast_cancer

print("Running breast cancer correlation matrix loop...")
start_correlation_loop_breast_cancer = time.time()
compute_correlation_naive(breast_cancer.data)
total_time_correlation_loop_breast_cancer = time.time() - start_correlation_loop_breast_cancer

print("Running breast cancer correlation matrix numpy...")
start_correlation_numpy_breast_cancer = time.time()
compute_correlation_smart(breast_cancer.data)
total_time_correlation_numpy_breast_cancer = time.time() - start_correlation_numpy_breast_cancer


digits = sklearn.datasets.load_digits()

print("Running digits pairwise loop...")
start_distance_loop_digits = time.time()
compute_distance_naive(digits.data)
total_time_distance_loop_digits = time.time() - start_distance_loop_digits


print("Running digits pairwise numpy...")
start_distance_numpy_digits = time.time()
compute_distance_smart(digits.data)
total_time_distance_numpy_digits = time.time() - start_distance_numpy_digits

print("Running digits correlation matrix loop...")
start_correlation_loop_digits = time.time()
compute_correlation_naive(digits.data)
total_time_correlation_loop_digits = time.time() - start_correlation_loop_digits

print("Running digits correlation matrix numpy...")
start_correlation_numpy_digits = time.time()
compute_correlation_smart(digits.data)
total_time_correlation_numpy_digits = time.time() - start_correlation_numpy_digits

distance_readings_loop = (total_time_distance_loop_iris, total_time_distance_loop_breast_cancer, total_time_distance_loop_digits)
distance_readings_cool = (total_time_distance_numpy_iris, total_time_distance_numpy_breast_cancer, total_time_distance_numpy_digits)
labels = ('Iris', 'Breast cancer', 'Digits')
N = len(distance_readings_loop)
width = 0.35
ind = np.arange(N)

fig, ax = plt.subplots()
rects1 = ax.bar(ind, distance_readings_loop, width, color='r')
rects2 = ax.bar(ind+width, distance_readings_cool, width, color='g', log=1)
ax.set_ylabel('Time (s)')
ax.set_title('Comparison distance computation methods')
ax.set_xticks(ind + width/2)
ax.set_xticklabels(labels)
ax.legend((rects1[0], rects2[0]), ('Loop implementation', 'Matrix implementation'))
plt.savefig('ComparisonBarDistances.pdf')

correlation_readings_loop = (total_time_correlation_loop_iris, total_time_correlation_loop_breast_cancer, total_time_correlation_loop_digits)
correlation_readings_cool = (total_time_correlation_numpy_iris, total_time_correlation_numpy_breast_cancer, total_time_correlation_numpy_digits)
labels = ('Iris', 'Breast cancer', 'Digits')
N = len(correlation_readings_loop)
width = 0.35
ind = np.arange(N)

fig, ax = plt.subplots()
rects1 = ax.bar(ind, correlation_readings_loop, width, color='r')
rects2 = ax.bar(ind+width, correlation_readings_cool, width, color='g', log=1)
ax.set_ylabel('Time (s)')
ax.set_title('Comparison correlation computation methods')
ax.set_xticks(ind + width/2)
ax.set_xticklabels(labels)
ax.legend((rects1[0], rects2[0]), ('Loop implementation', 'Matrix implementation'))
plt.savefig('ComparisonBarCorrelation.pdf')
if __name__ == "__main__": main()
