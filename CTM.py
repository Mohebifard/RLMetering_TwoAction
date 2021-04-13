import numpy as np

class simulation:

    def xyz(self, T, C, Cg, Cs, CI, Q, N, D, W, AC, X0, state_occupance = True, gate_signal=-1):
        
        # make X0 of the sink cells zero
        X0[Cs-1] = 0
        
        # make N of sink cells based on 150 time steps
        N[Cs-1] = 150 * Q[Cs[0]-1]
        
        Y = np.empty((len(AC), T-1))
        x = np.empty((T, len(C)))
        rho = 1
        
        if state_occupance:
            x[0] = np.array(X0*N)
        else:
            x[0] = np.array(X0)
        
        for t in range(int(T)-1):
            y = np.empty(len(AC));

            for i in range(len(AC)):
                
                I = int(AC[i, 0] - 1)
                J = int(AC[i, 1] - 1)
                Ind = int(AC[i, 2])

                if Ind == 0: # ordinary cells
                    y[i] = min(Q[I], Q[J], x[t, I], rho*(N[J] - x[t, J]))
                    continue
                
                if Ind == 2: # intersection cells
                    k = np.where(CI == AC[i, 0])[0]
                    y[i] = min(W[t, k] * Q[I], Q[J], x[t, I], rho*(N[J] - x[t, J]));
                    continue

                if Ind == 3: # diverge cells
                    diverge_links = np.where(AC[:, 0] == AC[i, 0])[0]
                    yd = min(Q[I], x[t, I])
                    
                    for link in diverge_links:
                        R = min( Q[int(AC[link, 1])-1], rho*(N[int(AC[link, 1])-1] - x[t, int(AC[link, 1])-1]))
                        yd = min(yd, R/AC[link, 3])
                    
                    y[i] = yd * AC[i, 3]
                    continue
                
                if Ind == 4: # merge cells
                    y[i] = min(Q[I], Q[J], x[t, I], rho*(N[J] - x[t, J]));
                    continue
                
                if Ind == 5: # ordinary cell before sink cell
                    y[i] = min(Q[I], Q[J], x[t, I], rho*(N[J] - x[t, J]))
                    continue

                if Ind == 1: # gate cells
                    y[i] = min(Q[I], Q[J], x[t, I], rho*(N[J] - x[t, J]))
                    
                    if gate_signal > -1:
                        k = np.where(Cg == AC[i, 0])[0]
                        y[i] = min(y[i], gate_signal * Q[I])            
                    
                    continue

            Y[:, t] = y[i]
            x[t + 1, :] = x[t, :]

            for i in range(len(AC)):
                I = int(AC[i, 0] - 1)
                J = int(AC[i, 1] - 1)
                Ind = int(AC[i, 2])
                
                if Ind == 1: # update gate cells
                    k = np.where(Cg == AC[i, 0])[0]
                    x[t + 1, I] += D[t, k] - y[i]
                    x[t + 1, J] += y[i]
                else: # all other cells
                    x[t + 1, I] += - y[i];
                    x[t + 1, J] += + y[i];
        
        return (x, Y);

