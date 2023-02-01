# imports
import numpy as np
import scipy.stats as stats
from easygui import msgbox


def t_test(mean1, mean2, SD1, SD2, num_points_1, num_points_2, alpha):
    """ do t test with the parameters mean, standard deviation and number of points """
    t = (mean1 - mean2) / (np.sqrt(np.power(SD1, 2) / num_points_1 + np.power(SD2, 2) / num_points_2))
    p_t = 2 * (stats.t.cdf(t, min(num_points_1, num_points_2) - 1))  # p(|t| > (value of t calculated above))
    if p_t < alpha:
        msgbox("Student's t-test result is " + str(p_t) + " ,which is smaller than your alpha " + str(
            alpha) + " ,thus we can reject the null hypothesis")
    else:
        msgbox("Student's t-test result is " + str(p_t) + " ,which is greater than your alpha " + str(
            alpha) + " ,thus we can not reject the null hypothesis")
