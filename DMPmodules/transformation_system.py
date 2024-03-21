import numpy as np

class TransformationSystem:

    def __init__(self, alpha_z, beta_z, cs):
        """
            Constructor of Transformation System
        """
        assert alpha_z > 0 and beta_z > 0, "alpha_z and beta_z must be positive"
        
        self.alpha_z = alpha_z
        self.beta_z = beta_z
        self.cs = cs
        self.tau = cs.tau

    def step(self, y_old, z_old, g, input, dt):
        """
           A forward integration of the differential equation for one step
        """
        assert dt > 0, "dt must be strictly positive"

        dy = z_old / self.tau
        dz = 1. / self.tau * (self.alpha_z * self.beta_z * (g - y_old) - self.alpha_z * z_old + input)

        y_new = y_old + dt * dy
        z_new = z_old + dt * dz

        return y_new, z_new, dy, dz

    def rollout( self, y0, z0, g, input_arr, t0i, t_arr ):
        """
            A full rollout of the trajectory
        """

        n  = len( y0    )
        Nt = len( t_arr )
        assert input_arr.shape == ( n, Nt - 1 ), "input_arr size must match"

        y_arr  = np.zeros( ( n, Nt ) )
        z_arr  = np.zeros( ( n, Nt ) )
        dy_arr = np.zeros( ( n, Nt ) ) 
        dz_arr = np.zeros( ( n, Nt ) ) 

        y_arr[  :, 0 ] = y0
        z_arr[  :, 0 ] = z0
        dy_arr[ :, 0 ] = z0 / self.tau

        for i in range( Nt - 1 ):

            if t_arr[i + 1] <= t0i:
                y_arr[ :, i + 1 ] = y0
                z_arr[ :, i + 1 ] = z0

            else:
                dt = t_arr[ i + 1] - t_arr[ i ]
                y_new, z_new, dy, dz = self.step( y_arr[:, i], z_arr[:, i], g, input_arr[:, i], dt )
                
                y_arr[  :, i + 1 ] = y_new
                z_arr[  :, i + 1 ] = z_new
                dy_arr[ :, i + 1 ] = dy
                dz_arr[ :, i + 1 ] = dz

        return y_arr, z_arr, dy_arr, dz_arr

    def get_desired(self, y_des, dy_des, ddy_des, g):
        """
            Calculate an array of f_des for Imitation Learning
        """
        # Convert scalar g to an array if necessary
        if np.isscalar(g):
            g = np.full(y_des.shape[0], g)

        # Ensure g is reshaped to match dimensions if it's not a scalar but has one dimension
        elif g.ndim == 1:
            g = g.reshape(-1, 1)
        
        # Check whether the 2D arrays of y_des, dy_des, ddy_des are all same size
        assert all(size == y_des.shape for size in [dy_des.shape, ddy_des.shape]), "All input arrays must have the same shape."
        assert g.shape[0] == y_des.shape[0], "The goal position array 'g' must have the same number of rows as 'y_des'."

        # Calculate the f_des array
        f_des = self.tau**2 * ddy_des + self.alpha_z * self.tau * dy_des + self.alpha_z * self.beta_z * (y_des - g)


        return f_des


