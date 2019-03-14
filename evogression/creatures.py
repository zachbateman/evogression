'''
Module containing "creatures" that each represent a potential regression equation.
'''
import random


class EvogressionCreature():


    def __init__(self,
                        full_parameter_example: dict,
                        target_parameter: str) -> None:

        # hunger decreases over time proportional to this creature's regression complexity
        # successfully "eating" a sample will increase self.hunger
        # creature dies when self.hunger == 0
        self.hunger = 100



    def __lt__(self, other):
        pass

    def __gt__(self, other):
        pass

    def __add__(self, other):
        pass

    def __repr__(self) -> str:
        return 'EvogressionCreature'

    def get_regression_func(self):
        return '...'
