
python -m cProfile -o "evolution_multip.profile" evogression_testing_multip.py

snakeviz evolution_multip.profile
