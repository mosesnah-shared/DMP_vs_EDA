import numpy as np

def gaussian( s_arr, ci, hi ):
    """
        Non-normalized Gaussian Function
        f(s) = exp(-hi(s-ci)^2)
    """
    return np.exp( -hi * ( s_arr - ci )**2 )

def von_mises( s_arr, ci, hi ):
    """
        Von-Mises Function
        f(s) = exp[hi{cos(s_arr - ci) - 1}]
    """
    return np.exp( hi * ( np.cos( s_arr - ci ) - 1) )

class NonlinearForcingTerm:
    def __init__( self, cs, N ):
        """
            Default Constructor of the Nonlinear Forcing Term
            fs = NonlinearForcingTerm(cs, N)
        """
        self.cs    = cs
        self.type  = self.cs.type
        self.N     = N
        self.c_arr = np.zeros( N )
        self.h_arr = np.zeros( N )
        
        # Discrete
        if self.cs.type == 0:  
            self.c_arr = np.exp(-self.cs.alpha_s / (N - 1) * np.arange(N))
            self.h_arr[ :-1 ] = 1.0 / np.diff( self.c_arr )**2
            self.h_arr[ -1  ] = self.h_arr[-2]

        # Rhythmic
        else:  
            self.c_arr = 2 * np.pi / N * np.arange( N )
            self.h_arr = 2.5 * N * np.ones( N )

    def calc_ith( self, t_arr, i ):
        """
            Calculate the activation of the i-th basis function at time t_arr
        """
        assert np.isscalar( t_arr ) or ( isinstance( t_arr, np.ndarray ) and t_arr.ndim == 1 )

        print( type( i ) )
        assert isinstance( i,  ( int, np.int32, np.int64 ) ) and 0 <= i <= self.N-1
        
        ci = self.c_arr[ i ]
        hi = self.h_arr[ i ]
        s_arr = self.cs.calc( t_arr )
        
        if self.type == 0:
            return  gaussian( s_arr, ci, hi )
        
        elif self.type == 1:
            return von_mises( s_arr, ci, hi )
        
        else:
            raise ValueError( "Wrong type for the Nonlinear Forcing Term" )

    def calc_multiple_ith( self, t_arr, i_arr ):
        """
            Calculate the activation of multiple i-th basis functions at time t, which can be either a scalar or an array
        """

        # If time array scalar, change to an array
        if np.isscalar( t_arr ):
            t_arr = np.array( [ t_arr ] )

        assert isinstance( t_arr, np.ndarray ) and t_arr.ndim == 1 
        assert isinstance( i_arr, np.ndarray ) and i_arr.ndim == 1 

        # Just in casee, sort the array
        i_arr.sort( )
        act_arr = np.zeros( ( len( i_arr ), len( t_arr ) ) )
        
        for i, idx in enumerate( i_arr ):
            act_arr[ i, : ] = self.calc_ith( t_arr, idx )
            
        return act_arr

    def calc_whole_at_t(self, t_arr):
        """
            Calculate the whole activation of the N basis functions at time t
        """
        return np.sum( self.calc_multiple_ith( t_arr, np.arange( self.N, dtype=int ) ), axis = 0) 

    def calc_whole_weighted_at_t(self, t_arr, w_arr):
        """
            Calculate the whole activation of the N basis with the weighting array w_arr
        """

        assert np.isscalar( t_arr ) or ( isinstance( t_arr, np.ndarray ) and t_arr.ndim == 1 )
        n, nc = w_arr.shape
        assert nc == self.N
        
        # If time array scalar, change to an array
        if np.isscalar( t_arr ):
            t_arr = np.array( [ t_arr ] )

        act_weighted = np.zeros( ( n, len( t_arr ) ) )
        
        for i in range( n ):
            act_weighted[ i, : ] = np.sum(w_arr[i, :].reshape(-1, 1) * self.calc_multiple_ith( t_arr, np.arange( self.N, dtype=int ) ) /
                                   self.calc_whole_at_t( t_arr ), axis = 0 )
            
        act_weighted[ np.isnan( act_weighted ) ] = 0
        return act_weighted

    def calc_forcing_term(self, t_arr, w_arr, t0i, scl, trimmed=False):
        """
        Calculate the nonlinear forcing term with a time offset of t0i.
        This function is for Imitation Learning.

        Parameters
        ----------
        t_arr : np.ndarray
            The time row array input.
        w_arr : np.ndarray
            The 2D Weight array to be multiplied.
            Shape: (n, N), where
                n: Number of dimensions of the Nonlinear Forcing Term,
                N: Number of basis functions.
        t0i : float
            The time offset for the nonlinear forcing term.
        scl : np.ndarray
            The 2D scaling square matrix.
            Must be the size of n x n.
        trimmed : bool, optional
            If True, applies trimming based on the canonical system's tau. Default is False.

        Returns
        -------
        force_arr : np.ndarray
            The n x Nt array, where Nt is the length of the time array.
        """
        assert t_arr.ndim == 1, "t_arr must be a 1-dimensional array"

        Nt = t_arr.size
        n, nc = w_arr.shape

        assert nc == self.N, "Number of columns in w_arr must equal the number of basis functions N"

        assert scl.shape == (n, n), "Scaling matrix scl must be of size n x n"
        
        # Discrete
        if self.cs.type == 0:
            assert t0i <= t_arr.max() and t0i >= 0, "t0i must be within the range of t_arr"
            
            force_arr = np.zeros((n, Nt))
            
            # Determine the indices for which the condition is True
            if trimmed:
                idx_arr = np.where((t_arr >= t0i) & (t_arr <= t0i + self.cs.tau))[0]
            else:
                idx_arr = np.where(t_arr >= t0i)[0]
            
            # Shift the indices by one to the right to include one step further from the integration
            idx_arr = np.roll(idx_arr, -1)
            
            # Calculate the forcing term
            t_tmp = t_arr[ idx_arr ] - t0i
            force_arr[:, idx_arr] = scl @ ( self.cs.calc( t_tmp ) * self.calc_whole_weighted_at_t( t_tmp, w_arr )  )
        
        # Rhythmic Movement
        else:
            force_arr = scl @ self.calc_whole_weighted_at_t(t_arr, w_arr)
        
        return force_arr