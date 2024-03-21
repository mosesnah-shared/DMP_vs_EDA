import numpy as np

class CanonicalSystem:
    def __init__(self, type, tau, alpha_s):
        """
        Default Constructor of the Canonical System
        cs = CanonicalSystem(type, tau, alpha_s)

        Parameters
        ----------
        (1) type : str
            'discrete' or 'rhythmic'

        (2) tau : float
            The time constant of the canonical system
            For 'discrete' movement: Duration of the movement
            For 'rhythmic' movement: Period of the movement divided by 2pi

        (3) alpha_s : float
            Positive constant, if 'rhythmic', then value is ignored

        """
        # Type input should be either 'discrete' or 'rhythmic'
        type = type.lower()
        assert type in [ "discrete", "rhythmic" ], "type must be 'discrete' or 'rhythmic'"

        # Discrete (0) or Rhythmic (1)
        self.type = 0 if type == "discrete" else 1

        # Tau and alpha_s should be positive values
        assert tau > 0 and alpha_s > 0, "tau and alpha_s must be positive values"

        self.tau = tau
        self.alpha_s = alpha_s

    def calc(self, t):
        """
        Calculating the Canonical System

        Parameters
        ----------
        (1) t : float or np.array
            Time (sec), accepts scalar or array input

        Returns
        -------
        (1) s : float or np.array
            The calculation of s(t)
            If discrete (0): s(t) = exp(-alpha_s/tau * t)
            If rhythmic (1): s(t) = mod(t/tau, 2pi)
        """

        if self.type == 0:
            # If Discrete
            s = np.exp(-self.alpha_s / self.tau * t)

        elif self.type == 1:
            # If Rhythmic
            s = np.mod( t / self.tau, 2 * np.pi)

        else:
            raise ValueError( f"[Wrong input] type should be either 0 or 1 but {self.type} is defined as type" )

        return s

# Example usage:
# cs = CanonicalSystem('discrete', 10, 1)
# print(cs.calc(1))
