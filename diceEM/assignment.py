from typing import List, Union, Tuple, Optional
import logging
import numpy as np
from numpy.typing import NDArray
from cse587Autils.DiceObjects.Die import Die, safe_exponentiate
from cse587Autils.DiceObjects.BagOfDice import BagOfDice

logger = logging.getLogger(__name__)

#<snip>
def dice_posterior(sample_draw: NDArray[np.int_], bag_of_dice: BagOfDice) -> float:
    """Calculates the posterior probability of a type 1 vs a type 2 die,
    based on the number of times each face appears in the draw, and the
    relative numbers of type 1 and type 2 dice in the bag, as well as the
    face probabilities for type 1 and type 2 dice. The single number returned
    is the posterior probability of the Type 1 die. Note: we expect a BagOfDice
    object with only two dice.

    :param sample_draw: The number of times each face appears in the draw
    :type sample_draw: list[int]
    :param type1Prior: The prior probability of a type 1 die
    :type type1Prior: float
    :param type2Prior: The prior probability of a type 2 die
    :type type2Prior: float
    :param faceProbs1: The probability of each face for a type 1 die
    :type faceProbs1: list[float]
    :param faceProbs2: The probability of each face for a type 2 die
    :type faceProbs2: list[float]
 
    :return: The posterior probability of a type 1 die
    :rtype: float
    """
    if len(bag_of_dice) > 2:
        raise ValueError('The bag of dice must contain only 2 dice')
    # note that this isn't strictly required by the bag_of_dice object, but
    # it is a assignment requirement. Requiring only two dice with the same
    # number of faces simplifies the problem a bit.
    if len(sample_draw) != len(bag_of_dice.dice[0]) or \
            len(sample_draw) != len(bag_of_dice.dice[1]):
        raise ValueError('The length of the sample_draw must be the same '
                         'length as both dice in the bag.')

    type_1_likelihood = 1.0
    type_2_likelihood = 1.0

    for i, num_faces in enumerate(sample_draw):
        type_1_likelihood *= safe_exponentiate(
            bag_of_dice.dice[0].face_probs[i], num_faces)
        type_2_likelihood *= safe_exponentiate(
            bag_of_dice.dice[1].face_probs[i], num_faces)

    type_1_posterior = type_1_likelihood * bag_of_dice.die_priors[0] / \
        (type_1_likelihood * bag_of_dice.die_priors[0] +
         type_2_likelihood * bag_of_dice.die_priors[1])

    return type_1_posterior

# </snip>

# Change to require that an initialized bag of dice be passed in.
# TODO check tests and eliminate any calls without a bag of dice argument.
def diceEM(experiment_data: List[NDArray[np.int_]],  # pylint: disable=C0103
           bag_of_dice: BagOfDice,
           accuracy: float = 1e-4,
           max_iterations: int = int(1e4)) -> Tuple[int, BagOfDice]:
    """
    Run the Expectation Maximization algorithm on roll results to
      estimate the parameters of the BagOfDice

    :param experiment_data: The results of repeatedly (with replacement)
        drawing a die from the bag and rolling a pre-determined number of
        times. The result of this trial is summarized by counting the number
        of times each face was rolled.
    :type experiment_data: list of numpy arrays with integer entries
    :param bag_of_dice: The initial BagOfDice object. This object stores the
        parameters of the dice in the bag.
    :type bag_of_dice: BagOfDice
    :param accuracy: The desired accuracy for the EM algorithm. When the
        difference between the parameters of the BagOfDice in two consecutive
        iterations is less than this value, the algorithm will terminate.
    :type accuracy: float
    :param calculate_likelihood: Whether to calculate the likelihood
        of the data given the parameters. The result is logged at level INFO
    :param max_iterations: The maximum number of iterations to run the
        algorithm for. If the algorithm does not converge before this number
        of iterations, it will terminate.
    :type max_iterations: int
    :rtype: tuple of (int, BagOfDice)

    """
    # check input types
    if not isinstance(bag_of_dice, BagOfDice):
        raise ValueError("bag_of_dice must be a BagOfDice object!")
    if not isinstance(accuracy, float) or accuracy <= 0:
        raise ValueError("accuracy must be a positive float!")
    if not isinstance(experiment_data, list):
        raise ValueError("experiment_data must be a list!")
    if not isinstance(max_iterations, int) or max_iterations <= 0:
        raise ValueError("max_iterations must be a positive integer")
    for roll in experiment_data:
        if not isinstance(roll, np.ndarray) or roll.dtype != np.int_:
            raise ValueError("Each element in experiment_data "
                             "must be a numpy ndarray!")
    
    # initialize a counter to keep track of the number of iterations
    iterations = 0
    # continue the E-M algorithm until the parameters converge to within the
    # desired accuracy or max_iterations has been reached. 
    while (((iterations == 0) or
            ((bag_of_dice - prev_bag_of_dice) > accuracy) and 
            (iterations < max_iterations))):
        # increment the number of iterations
        iterations += 1
        logging.debug("Iteration %s", iterations)

        # this is just for visualizing the progress of the algorithm
        logging.debug("Likelihood: %s",
                      bag_of_dice.likelihood(experiment_data))

        # E-step: compute the expected counts given current parameters
        # NOTE! you'll need to fill in the code here to call the
        # e_step() function
        expected_counts = None

        expected_counts = e_step(experiment_data, bag_of_dice)
        # M-step: update the parameters given the expected counts
        # NOTE! you'll need to fill in the code here to call the
        # m_step() function
        updated_bag_of_dice = m_step(expected_counts)

        # update the bag of dice objects for the next iteration
        # NOTE: This does make an assumption about the variable name you
        # use to store the output of the m_step() function. Feel free to
        # change this variable name if you want.
        prev_bag_of_dice: BagOfDice = bag_of_dice
        bag_of_dice = updated_bag_of_dice

    return iterations, bag_of_dice

def e_step(experiment_data: List[NDArray[np.int_]],
           bag_of_dice: BagOfDice) -> NDArray:
    """Performs the Expectation Step of the EM algorithm for the dice problem.

    Given a set of sample rolls and a current estimate of the bag of
        dice parameters, this function computes the expected counts of how
        many times each face of each die was rolled, based on the current
        dice parameters.

    :param experiment_data: A list of numpy arrays. Each array has length equal
        to the number of faces and records the number of times each face
        was rolled for a given draw. The number of arrays is equal to the 
        number of draws in the data.
    :type experiment_data: list of numpy arrays with integer entries
    :param bag_of_dice: A BagOfDice object. This object stores the
        parameters of the dice in the bag.
    :type bag_of_dice: BagOfDice

    :return: An array that is the same length as the number of dice in the bag.
        Each entry in the array is an array of floats which represent the
        expected number of times each face was rolled for the corresponding
        die.
    :rtype: np.array of np.arrays of floats
    """
    # Initialize the expected counts object for each die
    max_number_of_faces = max([len(die) for die in bag_of_dice.dice])
    # Initialize expected_counts to zero. It is a list of lists. The number
    # of inner lists is equal to the number of dice and the length of each
    # inner list is the number of faces of the die with the most faces.
    expected_counts = np.zeros((len(bag_of_dice), max_number_of_faces))

    # Iterate over draws. For each draw, calculate the the posterior probability
    # that each die type was rolled on that draw by calling dice_posterior.
    # Then combine the posterior for each die type with the observed counts for 
    # the current draw to get the expected counts for each die type on this draw.
    # To get the total expected counts for each type, you sum the expected
    # counts for each type over all the draws.  

    # PUT YOUR CODE HERE, FOLLOWING THE DIRECTIONS ABOVE

    # <snip>
    for draw in experiment_data:

        type_1_posterior = dice_posterior(draw, bag_of_dice)
        type_2_posterior = 1 - type_1_posterior

        for index, posterior in enumerate([type_1_posterior,
                                           type_2_posterior]):
            # Compute the expected number of faces
            expected_counts_this_draw_for_current_die = draw * posterior
            expected_counts[index] += expected_counts_this_draw_for_current_die
    # </snip>

    return expected_counts


def m_step(expected_counts_by_die: NDArray[np.float_]):
    """
    Performs the Maximization Step of the EM algorithm for the dice problem.
    It is called maximization because it performs maximum likelihood estimation
    of the hidden parameters.

    Given the expected counts of how many times each face of each die was
        rolled, this function computes the new parameters of the dice in the
        bag.

    :param expected_counts_by_die: An array that is the same length as the
        number of
        dice in the bag. Each entry in the array is an array of floats which
        represent the expected number of times each face was rolled for the
        corresponding die.
    :type expected_counts_by_die: A numpy array of numpy arrays of floats
    :return: A new BagOfDice object with updated parameters
    :rtype: BagOfDice
    """
    updated_type_1_frequency = np.sum(expected_counts_by_die[0])
    updated_type_2_frequency = np.sum(expected_counts_by_die[1])

    # REPLACE EACH NONE BELOW WITH YOUR CODE. 
    updated_priors = None
    updated_type_1_face_probs = None
    updated_type_2_face_probs = None
    # <snip>
    updated_priors = ([updated_type_1_frequency, updated_type_2_frequency] /
                      (updated_type_1_frequency + updated_type_2_frequency))

    updated_type_1_face_probs = (expected_counts_by_die[0] /
                                 updated_type_1_frequency)

    updated_type_2_face_probs = (expected_counts_by_die[1] /
                                 updated_type_2_frequency)
    # </snip>
    updated_bag_of_dice = BagOfDice(updated_priors,
                                    [Die(updated_type_1_face_probs),
                                     Die(updated_type_2_face_probs)])

    return updated_bag_of_dice