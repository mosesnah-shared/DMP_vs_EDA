import numpy as np

def data_diff(q_traj, t_arr):
    """
    Differentiate trajectory data with respect to time.

    Parameters:
    q_traj (numpy.ndarray): 2D array of trajectory data.
    t_arr (numpy.ndarray): 1D array of time points.

    Returns:
    numpy.ndarray: The derivative of the trajectory with respect to time.
    """
    # Ensure q_traj is a 2D numpy array and t_arr is a 1D numpy array
    assert q_traj.ndim == 2, "q_traj must be a 2D array."
    assert t_arr.ndim == 1, "t_arr must be a 1D array."

    # Assert that the data is appended along the column and matches the length of t_arr
    nr, nc = q_traj.shape
    assert nc > nr, "Data should be appended along the column."
    assert nc == len(t_arr), "Number of columns in q_traj must match the length of t_arr."

    # Differentiate and append the final point
    dq_traj = np.diff(q_traj, axis=1) / np.diff(t_arr)
    dq_traj = np.hstack((dq_traj, dq_traj[:, -1][:, np.newaxis]))  # Append the last column to maintain size

    return dq_traj
