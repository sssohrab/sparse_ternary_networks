from sparse_ternary_networks.Tools import *
import scipy.linalg as spla
from scipy.special import comb
#################### CLasses ####################################################################
class fwdPass_unit:
    """Performs a forward pass for one layer unit of the network."""
    def __init__(self,params,nlinParam,nlinStrategy='KBest_STC', meanExtraction='fromInput'):
        # input attributes:
        self.A = params['A']
        self.m, self.n = np.shape(self.A)
        try:self.mu = params['mu']
        except: self.mu = np.zeros((self.m,1))
        try:self.eta = params['eta']
        except:self.eta = np.zeros((self.n, 1))
        try:self.beta = params['beta']
        except:self.beta = np.ones((self.m,1))
        self.nlinParam = nlinParam
        self.nlinStrategy = nlinStrategy
        self.meanExtraction = meanExtraction
        # output attributes:
        self.dim_means = []
        self.prob_z = []
        self.entropy = []
        self.rate = []
        self.distortion = []
    def removeMean(self,F_in):
        """Removes the mean from dimensions. After the parameters are learned, it will be added in the end.

        If meanExtraction = 'fromInput', it will calculate the mean from the input to this class. If
        meanExtraction = 'off', it will perform neither removeMean() nor later addMean(). However,
         if meanExtraction is provided as a numpy array of correct dimension, it considers it as the given
         dim_means coming from some training set. So it performs the mean removal and addition based on this."""
        if isinstance(self.meanExtraction,str) == 1:
            if self.meanExtraction == 'fromInput':
                dim_means = np.mean(F_in, axis=1,keepdims=True)
                self.dim_means = dim_means
                F_in -= dim_means
            elif self.meanExtraction == 'off':
                pass
        else:
            dim_means = self.meanExtraction
            self.dim_means = dim_means
            F_in -= dim_means
        return F_in

    def encoder(self,F_in):
        """The encoder part to produce X, i.e., the Sparse Ternary Code."""
        #self.A /= np.linalg.norm(self.A,axis=1,keepdims=True)
        Z = self.A @ F_in + self.mu
        Phi = nlinearity(Z, self.nlinParam, self.nlinStrategy)
        X = Phi * self.beta
        assert np.count_nonzero(np.sum(np.abs(X), axis=0)) == X.shape[1]
        return X,Z,Phi

    def decoder(self,X):
        """The decoder part to reconstruct the F_in as F_hat.

        Remember this method should be essentially the same as BaseLearner_unit.decoder()."""
        F_hat = self.A.T @ X + self.eta
        return F_hat
    def addMean(self,F):
        """Adds the mean back to the data, i.e., both F_in anf F_hat."""
        if self.meanExtraction != 'off':
            F += self.dim_means
            return F

    def residue(self,F_in,F_hat):
        """The residual operator to build the input to the next stage F_out from F_in and F_hat.

        Remember this method should be exactly the same as BaseLearner_unit.residue()."""
        F_out = F_in - F_hat
        return F_out
    def entropyCalculator(self,X,ternaryProbMap='fromInput'):
        """Estimates the entropy and the overall rate of the ternary representation.

        To calculate the entropy, if ternaryProbMap='fromInput', it considers the given input as the training
        set to calculate the entropy probabilities. However, if an input is provided as the probability map
        learned from some training set, the entropy is calculated based on that ternary distribution learned
        from the training set, plus the deviation of these two distributions taken into account, i.e.,
        instead of H(test) we should calculate H(test) + D(test||train), where H() is the ternary entropy and
        D(||) is the KL-divergence btw. the two distributions.
        Note that the entropy is calculated only for the STC mode."""
        if  self.nlinStrategy.endswith('STC'):
            temp = np.zeros((np.shape(X)))
            temp[X == 0] = 1
            prob_z = np.divide(np.sum(temp, axis=1), temp.shape[1]).reshape(-1,1)
            alpha = 0.5 * (1 - prob_z)
            self.prob_z = prob_z
            if str(ternaryProbMap) == 'fromInput':
                self.entropy = ternary_entropy(alpha)
            else:
                alpha_train = 0.5*(1 - ternaryProbMap)
                D = ternary_KLDivergence(alpha, alpha_train)
                self.entropy = ternary_entropy(alpha) + D
            self.rate = np.divide(np.sum(self.entropy), self.n)
            if self.nlinStrategy.startswith('KBest'):
                if self.m > 10:
                    self.rate -= np.divide(
                        0.5 * np.log2(2 * np.pi * np.e * self.nlinParam *
                                      (1 - np.divide(self.nlinParam, self.m))), self.n)
                temp_rate_worstcase = np.divide(np.log2(
                    comb(self.m, self.nlinParam)) + self.nlinParam, self.n)
                self.rate = np.minimum(self.rate, temp_rate_worstcase)

    def distortionCalculator(self,F_out):
        """Calculates the normalized distortion of the reconstruction.

        The distortion is measured in l_2 sense and is normalized by both the dimension and the number of
        the samples of the input data."""
        distortion = np.divide(np.linalg.norm(F_out) ** 2, self.n * np.shape(F_out)[1])
        self.distortion = distortion

    def run(self,F_in):
        """Runs the methods in order, except for entropyCalculator()."""
        if self.meanExtraction != 'off':
            F_in = self.removeMean(F_in)
        X = self.encoder(F_in)[0] # Normally, actually, we're only interested in X.
        F_hat = self.decoder(X)
        if self.meanExtraction != 'off':
            F_in = self.addMean(F_in)
            F_hat = self.addMean(F_hat)
        F_hat = np.clip(F_hat, np.min(F_in), np.max(F_in))
        F_out = self.residue(F_in,F_hat)
        self.distortionCalculator(F_out)
        return F_out,F_hat,X
# ----------------------------------------------------------------------------------------------
class BaseLearner_unit:
    """Learns the parameters for one layer unit of the network given its input using back-prop free methods.

    """
    def __init__(self, nlinParam, m=None, nlinStrategy='KBest_STC', Learner='SuccessivePCA', meanExtraction = 'on', numSB = 1):
        # input attributes:
        self.nlinParam = nlinParam
        # if m is None:
        #     self.m = F_in.shape[0]
        # else:
        self.m = m
        self.n = None
        self.nlinStrategy = nlinStrategy
        self.Learner = Learner
        self.meanExtraction = meanExtraction
        self.numSB = numSB
        # output attributes:
        self.params = {}
        #self.X = []
        self.Sigma2 = []
        self.dim_means = []
        self.reWeight_coeff = []
        self.prob_z = []
        self.entropy = []
        self.rate = []
        self.distortion = []
    def removeMean(self,F_in):
        """Removes the mean from dimensions. After the parameters are learned, it will be added in the end."""
        if self.meanExtraction == 'on':
            dim_means = np.mean(F_in, axis=1,keepdims=True)
            self.dim_means = dim_means
            F_in -= dim_means
        return F_in
    def projMat(self,F_in):
        """Learns the projection matrix A, with PCA, identity or just random."""
        self.n = np.shape(F_in)[0]
        if self.m is None:
            self.m = self.n
        #
        if self.Learner == 'SuccessivePCA_obsolete':
            # This does not consider the idea of multi sub-band transformation.
            assert self.m <= self.n
            Cov_mat = np.cov(F_in)
            Sigma2, U = spla.eigh(Cov_mat)
            idx = Sigma2.argsort()[::-1]
            idx = idx[0:self.m]
            Sigma2 = Sigma2[idx[0:self.m]]
            U = U[:, idx[0:self.m]]
            # U = np.real_if_close(U,tol=1000)
            # U = np.real(U)
            self.Sigma2 = Sigma2.reshape(self.m,1)
            self.params['A'] = U.T
        elif self.Learner == 'SuccessivePCA' or self.Learner == 'ProcrusteanPCA':
            assert self.m <= self.n
            A = np.zeros((self.m, self.n))
            lenSB = np.floor(self.n/self.numSB)
            map1SB = np.append(np.arange(0, np.minimum(lenSB * self.numSB,self.m),
                                        lenSB), np.minimum(self.n,self.m))
            map2SB = np.append(np.arange(0, lenSB * self.numSB,lenSB), self.n)
            map1SB = map1SB.astype(int)
            map2SB = map2SB.astype(int)
            Sigma2_temp = np.array([])
            for sb in range(len(map1SB)-1):
                Cov_mat = np.cov(F_in[map2SB[sb]:map2SB[sb+1],:])
                Sigma2, U = spla.eigh(Cov_mat)
                idx = Sigma2.argsort()[::-1]
                Sigma2 = Sigma2[idx]
                U = U[:, idx]
                Sigma2_temp = np.hstack((Sigma2_temp,Sigma2))
                A[ map1SB[sb]:map1SB[sb+1], map2SB[sb]:map2SB[sb+1] ] = \
                    U[0:map2SB[sb+1]-map2SB[sb],0:map1SB[sb+1]-map1SB[sb]].T
            self.Sigma2 = Sigma2_temp[0:self.m].reshape(-1,1)
            assert np.shape(A) == (self.m,self.n)
            self.params['A'] = A
        elif self.Learner == 'Identity':
            self.params['A'] = np.eye(self.m,M=self.n)
            self.Sigma2 = np.var(F_in[0:self.m],axis=1,keepdims=True)
        elif self.Learner == 'Random':
            self.params['A'] = np.random.randn(self.m,self.n)*(np.divide(1,np.sqrt(self.m)))
            # Also assign mu and eta as zero vectors. This is not used here, but is compatible with
            # back-prop update classes.
        self.params['mu'] = np.zeros((self.m, 1))
        self.params['eta'] =  np.zeros((self.n, 1))
    def encoder(self,F_in):
        """Provides the code, i.e., X and specifies the optimal value for the weighting vector Beta."""
        if self.Learner == 'SuccessivePCA' or self.Learner == 'Identity'or self.Learner == 'SuccessivePCA_obsolete':
            if self.nlinStrategy.startswith('Threshold'):
                Phi = fwdPass_unit(self.params, self.nlinParam, 'Threshold').encoder(F_in)[2]
                equiv_thrsh = self.nlinParam
            elif self.nlinStrategy.startswith('KBest'):
                self.nlinParam = int(self.nlinParam)
                Phi = fwdPass_unit(self.params, self.nlinParam, 'KBest').encoder(F_in)[2]
                # This should be updated in the future. I know a better way to estimate this equivalent
                # threshold.
                equiv_thrsh = np.mean(np.sort(np.abs(Phi), axis=0)[self.m - self.nlinParam::][0])
            if self.nlinStrategy.endswith('STC'):
                Phi = np.sign(Phi)
                beta, Analytical_distortion = opt_ternary_alphabet(self.Sigma2,equiv_thrsh)
            else:
                beta = np.ones((self.m, 1))
            X = Phi * beta
            #assert np.count_nonzero(np.sum(np.abs(self.X), axis=0)) == self.X.shape[1]
            self.params['beta'] = beta
        elif self.Learner == 'Random':
            beta = np.ones((self.m,1))
            self.params['beta'] = beta
            X,Z,_ = fwdPass_unit(self.params, nlinParam=self.nlinParam,
                                 nlinStrategy=self.nlinStrategy).encoder(F_in)
            self.Sigma2 = np.var(Z, axis=1)
            assert np.count_nonzero(np.sum(np.abs(X), axis=0)) == F_in.shape[1]
        elif self.Learner == 'ProcrusteanPCA':
            # This part needs some clean-up. It's a bit redundant.
            num_iter = 100
            beta = np.ones((self.m, 1))
            self.params['beta'] = beta
            _, Z, _ = fwdPass_unit(self.params, nlinParam=self.nlinParam,
                                   nlinStrategy=self.nlinStrategy).encoder(F_in)
            R = np.random.randn(self.m, self.m) * (np.divide(1, np.sqrt(self.m)))
            #R = spla.orth(R)
            for iter in range(num_iter):
                tmpZ = R @ Z
                tmpPhi = nlinearity(tmpZ, self.nlinParam, self.nlinStrategy)
                # print(' ... ', iter, ' (', np.linalg.norm(tmpZ - tmpPhi)**2, ') ', end=' ')
                print(' -', end='')
                tmpU,_,tmpVT = spla.svd(tmpPhi@Z.T)
                R = tmpU @ tmpVT
            self.params['A'] = R @ self.params['A']
            X,Z,_ = fwdPass_unit(self.params, nlinParam=self.nlinParam,
                                  nlinStrategy=self.nlinStrategy).encoder(F_in)
            self.Sigma2 = np.var(Z, axis=1)
            # assert np.count_nonzero(np.sum(np.abs(X), axis=0)) == F_in.shape[1]
            # if self.nlinStrategy.startswith('Threshold'):
            #     Phi = fwdPass_unit(self.params, self.nlinParam, 'Threshold').encoder(F_in)[2]
            #     equiv_thrsh = self.nlinParam
            # elif self.nlinStrategy.startswith('KBest'):
            #     self.nlinParam = int(self.nlinParam)
            #     Phi = fwdPass_unit(self.params, self.nlinParam, 'KBest').encoder(F_in)[2]
            #     equiv_thrsh = np.mean(np.sort(np.abs(Phi), axis=0)[self.m - self.nlinParam::][0])
            # if self.nlinStrategy.endswith('STC'):
            #     Phi = np.sign(Phi)
            #     beta, Analytical_distortion = opt_ternary_alphabet(self.Sigma2,equiv_thrsh)
            # else:
            #     beta = np.ones((self.m, 1))
            # beta = beta.reshape(-1,1)
            # X = Phi * beta
            # #assert np.count_nonzero(np.sum(np.abs(self.X), axis=0)) == self.X.shape[1]
            # self.params['beta'] = beta

        return X
    def zeroCleanup(self,X):
        """ Cleans-up the apparently always inactive dimensions.

        After encoding, most particularly using PCA-based methods, if the sparsity level
        is very low w.r.t. the code-length m, some dimensions of the code might get no activity
        from all the examples of the training set. Then it makes sense to clean-up those dimensions
        from all the relevent attributes."""
        actvInd = np.sum(np.abs(X),axis=1) != 0
        X = X[actvInd,:]
        self.Sigma2 = self.Sigma2[actvInd]
        self.m = np.sum(actvInd == True)
        self.params['A'] = self.params['A'][actvInd,:]
        self.params['mu'] = self.params['mu'][actvInd]
        self.params['beta'] = self.params['beta'][actvInd]
        return X
    def decoder(self,X):
        """The decoder part to reconstruct the F_in as F_hat.

        Remember this method should be essentially the same as fwdPass_unit.decoder()"""
        F_hat = self.params['A'].T @ X
        return F_hat
    def reWeight(self,F_in,F_hat,X):
        """Multiplies the weighting vector beta by a coefficient learned from the data."""
        coeff_nume = 0.5*(np.linalg.norm(F_in)**2 + np.linalg.norm(F_hat)**2
        -np.linalg.norm(F_in - F_hat)**2)
        coeff_deno = np.linalg.norm(F_hat)**2
        reWeight_coeff = np.divide(coeff_nume,coeff_deno)
        #
        self.params['beta'] *= reWeight_coeff
        X *= reWeight_coeff
        F_hat *= reWeight_coeff
        self.reWeight_coeff = reWeight_coeff
        return F_hat,X

    def addMean(self,F_in):
        """Adds the mean back to the data."""
        if self.meanExtraction == 'on':
            F_in += self.dim_means
        return F_in
    def residue(self,F_in,F_hat):
        """The residual operator to build the input to the next stage F_out from F_in and F_hat.

        Remember this method should be exactly the same as fwdPass_unit.residue()"""
        F_out = F_in - F_hat
        return F_out

    def entropyCalculator(self,X):
        """Estimates the entropy and the overall rate of the ternary representation.

        Uses the statistics of the encoded training data to estimate the probabilities of the ternary elements
        (assuming symmetry) for each of its dimensions. Based on this, the entropy for each dimension along
        with the overall rate is calculated.
        Note that the entropy is calculated only for the STC mode."""
        if self.nlinStrategy.endswith('STC'):
            temp = np.zeros((np.shape(X)))
            temp[X == 0] = 1
            prob_z = np.divide(np.sum(temp, axis=1), temp.shape[1]).reshape(-1,1)
            alpha = 0.5 * (1 - prob_z)
            self.prob_z = prob_z
            self.entropy = ternary_entropy(alpha)
            self.rate = np.divide(np.sum(self.entropy), self.n)
            if self.nlinStrategy.startswith('KBest'):
                if self.n > 10:
                    self.rate -= np.divide(
                        0.5 * np.log2(2 * np.pi * np.e * self.nlinParam *
                                      (1 - np.divide(self.nlinParam, self.m))), self.n)
                temp_rate_worstcase = np.divide(np.log2(
                                           comb(self.m,self.nlinParam)) + self.nlinParam,self.n)
                self.rate = np.minimum(self.rate,temp_rate_worstcase)

    def distortionCalculator(self,F_out):
        """Calculates the normalized distortion of the reconstruction.

        The distortion is measured in l_2 sense and is normalized by both the dimension and the number of
        the samples of the input data."""
        distortion = np.divide(np.linalg.norm(F_out)**2,self.n*F_out.shape[1])
        self.distortion = distortion

    def run(self,F_in):
        """Runs all the methods of this class in order."""
        if self.meanExtraction == 'on':
            F_in = self.removeMean(F_in)
        self.projMat(F_in)
        X = self.encoder(F_in)
        X = self.zeroCleanup(X)
        F_hat = self.decoder(X)
        F_hat,X = self.reWeight(F_in,F_hat,X)
        if self.meanExtraction == 'on':
            F_in = self.addMean(F_in)
            F_hat = self.addMean(F_hat)
        F_hat = np.clip(F_hat, np.min(F_in), np.max(F_in))
        F_out = self.residue(F_in,F_hat)
        self.entropyCalculator(X)
        self.distortionCalculator(F_out)
        return F_out,F_hat,X

    def reRun_weightUpdate(self,F_in,params):
        """
        In some scenarios, A, mu and eta are provided, but beta needs to be re-calculated.

        This is the same as run method, but without learning the projectors.
        """
        self.params = params
        # self.Learner = 'SuccessivePCA'
        self.m = np.shape(self.params['A'])[0]
        #self.params['beta'] = np.ones((self.m,1))  # Just momentarily. It will be re-calculated shortly.
        if self.meanExtraction == 'on':
            F_in = self.removeMean(F_in)

        Z = fwdPass_unit(self.params, self.nlinParam, self.nlinStrategy).encoder(F_in)[1]
        self.Sigma2 = np.var(Z, axis=1).reshape(-1,1)
        X = self.encoder(F_in)
        X = self.zeroCleanup(X)
        F_hat = self.decoder(X)
        F_hat,X = self.reWeight(F_in,F_hat,X)
        if self.meanExtraction == 'on':
            F_in = self.addMean(F_in)
            F_hat = self.addMean(F_hat)
        F_hat = np.clip(F_hat, np.min(F_in), np.max(F_in))
        F_out = self.residue(F_in,F_hat)
        return F_out,F_hat,X
    def generate(self,num_gen):
        """Generates random codes like the corresponding codes for the training samples.

        This method is not called from obj.run(). It should be called separately by specifying the
        desired number of codes to be generated.
        Note that this is meaningful only in the ternary cases, i.e., nlinStrategy = 'KBest_STC' or
        nlinStrategy = 'Threshold_STC'"""
        assert self.nlinStrategy.endswith('STC') == True
        X_gen = np.array([]).reshape(0, num_gen)
        for mInd in range(self.m):
            prob_alphabet = np.hstack([0.5*(1-self.prob_z[mInd]),
                                       self.prob_z[mInd], 0.5*(1-self.prob_z[mInd])])
            x = np.random.choice([-1, 0., 1], size=(1, num_gen), p=prob_alphabet)
            X_gen = np.vstack((X_gen, x))
        X_gen *= self.params['beta']
        F_gen = self.params['A'].T@X_gen
        if self.meanExtraction == 'on':
            F_gen += self.dim_means
        return X_gen,F_gen


# ----------------------------------------------------------------------------------------------
class fwdPass:
    """Performs one forward pass through the entire network.

    This is based on the "fwdPass_unit" class as building block of each layer unit."""
    def __init__(self,params,nlinParam,nlinStrategy='KBest_STC',ternaryProbMap='fromInput',meanExtraction='fromInput'):
        # input attributes:
        self.params = params
        self.L = len(params)
        if type(nlinParam) is list:
            assert len(nlinParam) == self.L
            self.nlinParam = nlinParam
        else:
            self.nlinParam = [nlinParam]*self.L
        if type(nlinStrategy) is list:
            assert len(nlinStrategy) == self.L
            self.nlinStrategy = nlinStrategy
        else:
            self.nlinStrategy = [nlinStrategy]*self.L
        if type(ternaryProbMap) is list:
            assert len(ternaryProbMap) == self.L
            self.ternaryProbMap = ternaryProbMap
        else:
            self.ternaryProbMap = [ternaryProbMap]*self.L
        if type(meanExtraction) is list:
            assert len(meanExtraction) == self.L
            self.meanExtraction = meanExtraction
        else:
            self.meanExtraction = [meanExtraction]*self.L
        # output attributes:
        self.F_hat = []
        self.prob_z = []
        self.entropy = []
        self.rate = []
        self.distortion = []
    def run(self,F_in):
        """Sequentially running the layer units."""
        F_hat = []
        X = []
        self.distortion.append(np.divide(np.linalg.norm(F_in) ** 2,np.prod(F_in.shape)))
        self.rate.append(0.)
        print(' Running the network:')
        print(' ##################  Running the network:  ##################')
        print('layer-units:')
        for l in range(self.L):
            print('**', l + 1, '**', end='')
            print('\n' * ((l + 1) % 15 == 0), end='')
            obj = fwdPass_unit(self.params[l],self.nlinParam[l],self.nlinStrategy[l],self.meanExtraction[l])
            F_out,F_hat_tmp,X_tmp = obj.run(F_in)
            F_hat.append(F_hat_tmp)
            self.distortion.append(obj.distortion)
            obj.entropyCalculator(X_tmp,self.ternaryProbMap[l])
            self.prob_z.append(obj.prob_z)
            self.entropy.append(obj.entropy)
            self.rate.append(obj.rate)
            X.append(X_tmp)
            F_in = F_out
        print('\nFinished running the network.')
        if self.nlinStrategy[0].endswith('STC'):
            self.rate = np.array(np.cumsum(self.rate))
        else:
            self.rate = np.array([])
        self.distortion = np.array(self.distortion)
        return F_out,F_hat,X


# ----------------------------------------------------------------------------------------------
class BaseLearner:
    """Learns the parameters of the entire network using back-prop free methods.

    This is based on the "BaseLearner_unit" class as building block of each layer unit."""
    def __init__(self, nlinParam, L,m=None, nlinStrategy='KBest_STC', Learner='SuccessivePCA', numSB = 1, meanExtraction='on'):
        # input attributes:
        self.L = L
        if type(m) is list:
            assert len(m) == self.L
            self.m = m
        else:
            self.m = [m]*self.L
        if type(nlinParam) is list:
            assert len(nlinParam) == self.L
            self.nlinParam = nlinParam
        else:
            self.nlinParam = [nlinParam]*self.L
        if type(nlinStrategy) is list:
            assert len(nlinStrategy) == self.L
            self.nlinStrategy = nlinStrategy
        else:
            self.nlinStrategy = [nlinStrategy]*self.L
        if type(Learner) is list:
            assert len(Learner) == self.L
            self.Learner = Learner
        else:
            self.Learner = [Learner]*self.L
        if type(numSB) is list:
            assert len(numSB) == self.L
            self.numSB = numSB
        else:
            self.numSB = [numSB]*self.L
        if type(meanExtraction) is list:
            assert len(meanExtraction) == self.L
            self.meanExtraction = meanExtraction
        else:
            self.meanExtraction = [meanExtraction]*self.L
        # output attributes:
        self.params = []
        self.entropy = []
        self.rate = []
        self.distortion = []
        self.prob_z = []
        self.dim_means = []
        self.Sigma2 = []


    def run(self,F_in):
        """Sequentially running the layer units."""
        F_hat = []
        X = []
        self.distortion.append(np.divide(np.linalg.norm(F_in) ** 2,np.prod(F_in.shape)))
        self.rate.append(0.)
        print(' ################## Starting to learn network parameters:  ##################')
        print('layer-units:')
        for l in range(self.L):
            print('**', l + 1, '**', end='')
            print('\n' * ((l + 1) % 15 == 0), end='')
            obj = BaseLearner_unit(self.nlinParam[l],m=self.m[l],
                                   nlinStrategy=self.nlinStrategy[l],Learner=self.Learner[l],
                                   numSB=self.numSB[l],meanExtraction=self.meanExtraction[l])
            F_out,F_hat_tmp,X_tmp = obj.run(F_in)
            self.params.append(obj.params)
            F_hat.append(F_hat_tmp)
            self.distortion.append(obj.distortion)
            self.rate.append(obj.rate)
            F_in = F_out
            self.prob_z.append(obj.prob_z)
            self.entropy.append(obj.entropy)
            X.append(X_tmp)
            self.dim_means.append(obj.dim_means)
            self.m[l] = obj.m
            self.Sigma2.append(obj.Sigma2)
        print('\nFinished learning network parameters:')
        if self.nlinStrategy[0].endswith('STC'):
            self.rate = np.array(np.cumsum(self.rate))
        else:
            self.rate = np.array([])
        self.distortion = np.array(self.distortion)
        return F_out,F_hat,X

    def reRun_weightUpdate(self,F_in,params):
        """Similar to run, but without learning the projectors, just beta."""
        F_hat = []
        X = []
        self.params = []
        print(' ################## Starting to learn network parameters:  ##################')
        print('layer-units:')
        for l in range(self.L):
            print('**', l + 1, '**', end='')
            print('\n' * ((l + 1) % 15 == 0), end='')
            obj = BaseLearner_unit(self.nlinParam[l], m=self.m[l],
                                   nlinStrategy=self.nlinStrategy[l], Learner=self.Learner[l],
                                   numSB=self.numSB[l], meanExtraction=self.meanExtraction[l])
            F_out, F_hat_tmp, X_tmp = obj.reRun_weightUpdate(F_in,params[l])
            self.params.append(obj.params)
            self.m.append(obj.m)
            F_in = F_out
        print('\nFinished learning network parameters:')
        return F_out, F_hat, X

    def generate(self,num_gen):
        """
        Generates random codes like the corresponding codes for the training samples.

        This is the multi-layer version of BaseLearner_unit.generate(num_gen).
        Note: I need to check this method again. After re-structuring the classes,
        I have yet not checked weather this works.
        """

        print(' ....Generating random codes.')
        F_gen = []
        X_gen = []
        for l in range(self.L):
            assert self.nlinStrategy[l].endswith('STC') == True
            m = self.m[l]
            X_gen_tmp = np.array([]).reshape(0, num_gen)
            for mInd in range(m):
                prob_alphabet = np.hstack([0.5 * (1 - self.prob_z[l][mInd]),
                                       self.prob_z[l][mInd], 0.5 * (1 - self.prob_z[l][mInd])])
                x = np.random.choice([-1, 0., 1], size=(1, num_gen), p=prob_alphabet)
                X_gen = np.vstack((X_gen, x))
            X_gen_tmp *= self.params[l]['beta']
            X_gen.append(X_gen_tmp)
            F_gen_tmp = self.params[l]['A'].T @ X_gen
            if self.meanExtraction[l] == 'on':
                F_gen_tmp += self.dim_means[l]
            F_gen.append(F_gen_tmp)
        return X_gen,F_gen
# ----------------------------------------------------------------------------------------------