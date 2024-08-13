import logging
import numpy as np
import unittest
from gradescope_utils.autograder_utils.decorators import weight #type: ignore
from cse587Autils.DiceObjects.BagOfDice import BagOfDice
from cse587Autils.DiceObjects.Die import Die
from diceEM.assignment import e_step, m_step, diceEM
from diceEM.utils.configure_logging import configure_logging

logger = logging.getLogger(__name__)

configure_logging(logging.INFO)


# NOTE: [1/6] * 6 may look like it would equal 1. However, in python this 
# is a shorthand method of creating a list of 6 repeated elements of value 
# 1/6


class TestDiceEM(unittest.TestCase):
    """
    Test the diceEM function. Note to students: This test file, exactly 
    as you first see it, are the tests that will be used to autograde 
    your assignment. You can run these tests yourself to see how you are
    doing. If you make any changes, just remember that this changes WILL NOT 
    be used to grade your assignment.
    """

    def test_e_step(self):
        """
        This test is not graded. Instead, it is provided for you to help
        debug your code.
        """
        experiment_data = np.array(
            [[10, 10, 10, 10, 10, 10] for _ in range(100)])

        # Test with equal probabilities. Since the dice are identical,
        # the data provides no information about which die was rolled,
        # so the posterior is equal to the prior -- prob of each die is 0.5.
        # Thus, if face 2 is rolled 10 times on a given draw, 5 of those
        # rolls are attributed to each die. The data contiains 1000 total rolls 
        # of each face, so 500 are attributed to each die. 
        bag_of_dice = BagOfDice(
            [0.5, 0.5], [Die([1 / 6] * 6), Die([1 / 6] * 6)])
        actual_1 = e_step(experiment_data, bag_of_dice)
        expected_1 = np.array([[500.0] * 6, [500.0] * 6])
        np.testing.assert_almost_equal(actual_1, expected_1, decimal=5)

        # Test with different face probabilities. The first die should get
        # more expected counts because its face probabilities (all equal) are
        # we expect that the first die will have more expected counts are a
        # better match for the observed face counts (all equal).
        bag_of_dice = BagOfDice(
            [0.5, 0.5], [Die([1 / 6] * 6), Die([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])]
        )
        actual_2 = e_step(experiment_data, bag_of_dice)
        sums = np.sum(actual_2, axis=1)
        self.assertTrue(sums[0] > sums[1])

    def test_m_step(self):
        """
        This is not a graded test. It is provided for you to help debug your code.
        """
        # Generate some random expected counts
        expected_counts = np.random.uniform(0, 1, (2, 6))
        # Make sure that expected counts for each die sum to a positive number
        expected_counts /= np.sum(expected_counts, axis=1, keepdims=True)
        expected_counts *= np.random.uniform(1, 10, (2, 1))

        new_bag_of_dice = m_step(expected_counts)

        # The new dice should have the same number of faces as the input
        self.assertTrue(all([len(die[1]) == 6 for die in new_bag_of_dice]))
        # The new priors should be proportional to the total count for each die
        self.assertTrue(np.allclose(
            [prior for prior, _ in new_bag_of_dice],
            np.sum(expected_counts, axis=1) / np.sum(expected_counts),
            atol=1e-2,
        ))
        # For each die, the face probabilities should be proportional to the
        # expected counts
        for die, expected_face_counts in zip(new_bag_of_dice, expected_counts):
            self.assertTrue(np.allclose(
                die[1], expected_face_counts /
                np.sum(expected_face_counts), atol=1e-2))
    @weight(6)
    def test_1(self):
        """
        This test simply checks that the e_step produces the correct output
        type and shape.
        """
        experiment_data = np.array(
            [[10, 10, 10, 10, 10, 10] for _ in range(100)])

        # Test with equal probabilities
        bag_of_dice = BagOfDice([0.5, 0.5],
                                [Die([1 / 6] * 6), 
                                 Die([1 / 6] * 6)])
        actual_1 = e_step(experiment_data, bag_of_dice)

        self.assertEqual(actual_1.shape, (2, 6))
        
    @weight(6)
    def test_2(self):
        """
        This test simply checks that the m_step produces the correct output
        type and shape.
        """
        # Generate some random counts
        random_counts = np.random.uniform(0, 1, (2, 6))

        bag_of_dice = m_step(random_counts)

        self.assertIsInstance(bag_of_dice, BagOfDice)
        self.assertEqual(len(bag_of_dice), 2)

    @weight(6)
    def test_3(self):
        """Original diceEM test 1

        Here there are clearly two different types of dice, one that only
        rolls face 0 and one that only rolls face 2.
        The die type that rolls face 1 was drawn on 3 trials and the type that
        rolls face 2 was drawn on 1 trial.
        The correct solution is found in three iterations.
        """

        experiment_data = [[10, 0, 0], [0, 0, 10], [10, 0, 0], [10, 0, 0]]

        experiment_data = [np.array(x) for x in experiment_data]

        initial_bag = BagOfDice(
            [0.45, 0.55], [Die([0.3, 0.25, 0.45]), Die([0.5, 0.3, 0.2])]
        )

        actual_num_iterations, estimated_bag_of_dice = diceEM(
            experiment_data, initial_bag, accuracy=1e-4)
        estimated_bag_of_dice.sort()

        expected_bag = BagOfDice([0.25, 0.75],
                                 [Die([0.0, 0.0, 1.0]),
                                 Die([1.0, 0.0, 0.0])])

        self.assertTrue(estimated_bag_of_dice - expected_bag < 1e-100)
        logger.info("actual_num_iterations: %s", actual_num_iterations)

    @weight(6)
    def test_4(self):
        """Original diceEM test 2

        In the previous example, if the draw on which the rare die type is \
        chosen also has examples of face 1, then the inferred face probs for 
        that die should reflect that . But the examples of face 2 provide 
        no new information about which die was drawn on each trial, so nothing 
        else should change . Again the correct solution (to two decimal places) 
        is found in 2 iterations, although it takes 3 until the convergence 
        criterion provided is met .
        """

        experiment_data = [[15, 0, 0], [0, 5, 10], [15, 0, 0], [15, 0, 0]]
        experiment_data = [np.array(x) for x in experiment_data]

        initial_bag = BagOfDice(
            [0.45, 0.55], [Die([0.3, 0.25, 0.45]), Die([0.5, 0.3, 0.2])]
        )

        actual_num_iterations, estimated_bag_of_dice = diceEM(
            experiment_data, initial_bag, accuracy=1e-4)
        estimated_bag_of_dice.sort()

        expected_bag = BagOfDice([0.25, 0.75],
                                 [Die([0.0, 1/3, 2/3]),
                                 Die([1.0, 0.0, 0.0])])

        self.assertTrue(estimated_bag_of_dice - expected_bag < 1e-100)
        logger.info("actual_num_iterations: %s", actual_num_iterations)

    @weight(6)
    def test_5(self):
        """Original diceEM test 3

        Even when the initial face probes are very similar and the convergence
        criterion is much tighter, the correct solution is found in 4 iterations.

        """
        experiment_data = [[15, 0, 0], [0, 5, 10], [15, 0, 0], [15, 0, 0]]
        experiment_data = [np.array(x) for x in experiment_data]

        initial_bag = BagOfDice(
            [0.45, 0.55], [Die([0.3, 0.25, 0.45]), Die([0.25, 0.3, 0.45])]
        )

        actual_num_iterations, estimated_bag_of_dice = diceEM(
            experiment_data, initial_bag, accuracy=1e-10)
        estimated_bag_of_dice.sort()

        expected_bag = BagOfDice([0.25, 0.75],
                                 [Die([0.0, 1/3, 2/3]),
                                 Die([1.0, 0.0, 0.0])])

        logger.info("actual_num_iterations: %s", actual_num_iterations)
        #print("actual_num_iterations: %s", actual_num_iterations)
        self.assertTrue(actual_num_iterations == 4)
        self.assertTrue(estimated_bag_of_dice - expected_bag < 1e-100)

    @weight(6)
    def test_6(self):
        """original diceEM test 4

        Here both die types can generate face 2. That should be reflected in
            the inferred face probs for die type 2,but it shouldn't 
            change anything else. Convergence in 4 iterations.
        """
        experiment_data = [[15, 0, 0], [0, 5, 10], [15, 0, 0], [10, 5, 0]]
        experiment_data = [np.array(x) for x in experiment_data]

        # Initial bag setup from the Mathematica test
        initial_bag = BagOfDice(
            [0.45, 0.55], [Die([0.3, 0.2, 0.5]), Die([0.25, 0.3, 0.45])]
        )

        # Perform the diceEM function with the given experiment_data and
        # initial_bag
        actual_num_iterations, estimated_bag_of_dice = diceEM(
            experiment_data, initial_bag, accuracy=1e-10)
        estimated_bag_of_dice.sort()

        # Expected bag results from the Mathematica test
        expected_bag = BagOfDice([0.25, 0.75],
                                 [Die([0.0, 1/3, 2/3]),
                                 Die([0.89, 0.11, 0.0])])

        logger.info("actual_num_iterations: %s", actual_num_iterations)
        # Assert that the difference between the estimated
        # and expected bags is very small
        # NOTE: the result comes out to .88888888...9 and .11111....
        print("actual_num_iterations: %s", actual_num_iterations)
        self.assertTrue(estimated_bag_of_dice - expected_bag < 1e-2)

    @weight(6)
    def test_7(self):
        """original diceEM test 5

        Here it is much less clear which die type is rolled each time.
            The intuitive solution is found,
            with 25 % one die type, 75 % the other, and symmetrical face probs
            (one die the reverse of the other. But it takes more iterations.
        """
        # Sample rolls from the new Mathematica test
        experiment_data = [[8, 5, 2], [2, 5, 8], [8, 5, 2], [8, 5, 2]]
        experiment_data = [np.array(x) for x in experiment_data]

        # Initial bag setup remains consistent with previous tests
        initial_bag = BagOfDice(
            [0.45, 0.55], [Die([0.3, 0.2, 0.5]), Die([0.25, 0.3, 0.45])]
        )

        # Perform the diceEM function with the given experiment_data and
        # initial_bag
        actual_num_iterations, estimated_bag_of_dice = diceEM(
            experiment_data, initial_bag, accuracy=1e-10)
        estimated_bag_of_dice.sort()

        # Expected bag results from the new Mathematica test
        expected_bag = BagOfDice([0.25, 0.75],
                                 [Die([0.13666666666666671, 1/3, 0.53]),
                                 Die([0.53, 1/3, 0.13666666666666671])])

        logger.info("actual_num_iterations: %s", actual_num_iterations)
        self.assertTrue(estimated_bag_of_dice - expected_bag < 1e-1)

    @weight(6)
    def test_8(self):
        """
        Here the different types of trials are less distinguishable and
            the algorithm decides that the 
            there is non-negligeable probability that either type of trial
            could have been generated by 
            either type of die. It also takes much longer to converge.
        """
        # Sample rolls from the new Mathematica test
        experiment_data = [[7, 5, 3], [3, 5, 7], [7, 5, 3], [7, 5, 3]]
        experiment_data = [np.array(x) for x in experiment_data]

        # Initial bag setup remains consistent with previous tests
        initial_bag = BagOfDice(
            [0.45, 0.55], [Die([0.3, 0.2, 0.5]), Die([0.25, 0.3, 0.45])]
        )

        # Perform the diceEM function with the given experiment_data and
        # initial_bag
        actual_num_iterations, estimated_bag_of_dice = diceEM(
            experiment_data, initial_bag, accuracy=1e-10)
        estimated_bag_of_dice.sort()

        # Expected bag results from the new Mathematica test
        expected_bag = BagOfDice([0.23, 0.77],
                                 [Die([0.21666666666666667, 1/3, 0.45]),
                                 Die([0.45, 1/3, 0.21666666666666667])])

        logger.info("actual_num_iterations: %s", actual_num_iterations)
        # difference is ~0.028
        self.assertTrue(estimated_bag_of_dice - expected_bag < 1e-1)