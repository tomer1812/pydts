import os

PROJECT_DIR = os.path.join(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, '../../output')
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
