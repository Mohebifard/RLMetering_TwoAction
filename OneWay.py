import numpy as np

class info:
    def __init__(self):
        self.T = 300
        self.C = np.array( [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,])
        self.Cg = np.array([1,14,27,40,])
        self.Cs = np.array([13,26,39,52])
        self.CI = np.array([4,34,35,8,9,43,47,48,17,21,22,30,]) 
        self.Cm = np.array([5,36,10,44,49,18,23,31,])
        self.Cd = np.array([33,7,46,20])
        self.Q = np.array([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,])
        self.N = np.array([1000,12,12,12,12,12,12,12,12,12,12,12,1000,1000,12,12,12,12,12,12,12,12,12,12,12,1000,1000,12,12,12,12,12,12,12,12,12,12,12,1000,1000,12,12,12,12,12,12,12,12,12,12,12,1000,])
        self.AC = np.array([[1, 2, 1, 1], [2, 3, 0, 1], [3, 4, 0, 1], [4, 5, 2, 1], [5, 6, 4, 1], [6, 7, 0, 1], [7, 8, 3, 0.6], [7, 9, 3, 0.4], [8, 44, 2, 1], [9, 10, 2, 1], [10, 11, 4, 1], [11, 12, 0, 1], [12, 13, 5, 1], [14, 15, 1, 1], [15, 16, 0, 1], [16, 17, 0, 1], [17, 18, 2, 1], [18, 19, 4, 1], [19, 20, 0, 1], [20, 21, 3, 0.6], [20, 22, 3, 0.4], [21, 31, 2, 1], [22, 23, 2, 1], [23, 24, 4, 1], [24, 25, 0, 1], [25, 26, 5, 1], [27, 28, 1, 1], [28, 29, 0, 1], [29, 30, 0, 1], [30, 31, 2, 1], [31, 32, 4, 1], [32, 33, 0, 1], [33, 34, 3, 0.6], [33, 35, 3, 0.4], [34, 5, 2, 1], [35, 36, 2, 1], [36, 37, 4, 1], [37, 38, 0, 1], [38, 39, 5, 1], [40, 41, 1, 1], [41, 42, 0, 1], [42, 43, 0, 1], [43, 44, 2, 1], [44, 45, 4, 1], [45, 46, 0, 1], [46, 47, 3, 0.6], [46, 48, 3, 0.4], [47, 18, 2, 1], [48, 49, 2, 1], [49, 50, 4, 1], [50, 51, 0, 1], [51, 52, 5, 1]])
        self.D = np.array([[3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],[  3.0,  3.0,  3.0,  3.0,],])
        self.W = np.array([[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],[1,0,0,1,1,0,0,0,1,1,1,0,],[0,1,1,0,0,1,1,1,0,0,0,1,],])
        self.X0= 0*np.ones(len(self.C))
        
        