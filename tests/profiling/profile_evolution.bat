
python -m cProfile -o "evolution.profile" evogression_testing.py

snakeviz evolution.profile
