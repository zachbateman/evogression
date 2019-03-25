# Evogression

Evogression is Python package providing an evolutionary algorithm to develop a regression function for a target parameter.  An arbitrary number of input parameters can be used, and data samples need not have all parameters populated to be used in training the regression.

# Current Features

  - EvogressionCreature class provides randomized regression functions given an parameter dictionary.  These "creatures" can each predict an output value given an input parameter dictionary either from training data or for new predictions.
  - CreatureEvolution class generates a group of creatures which then compete to most accurately model the training data.  Creatures are rewarded when they are better at modeling results than their peers, and a survival-of-the-fittest situation emerges.  Creatures are "mated" with resulting offspring having a combination of the parent creatures' regression characteristics along with some mutation.
  - CreatureEvolution class uses the Standardizer class to standardize input data.  This allows for regression parameters that estimate and evolve efficiently regardless of data scaling.
  - Cycles of "feast" and "famine" cause the community of creatures to grow and shrink with each phase either increasing the diversity of creatures (regression equations) or decreasing the diversity by killing off the lower-performing creatures.

# Testing

  - Brute force generation of creatures successfully models linear and parabolic 2D data.

<img src="tests/images/linear_regression_single_layer_brute_force_test.png">

  - Evolution algorithm successfully models linear and parabolic 2D data.  Additional dimensions of input data are still to be tested.

<img src="tests/images/parabola_regression_evolution_test.png">



License
----
TODO
