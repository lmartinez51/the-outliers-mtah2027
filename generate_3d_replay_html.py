"""Root forwarder for src/generate_3d_replay_html.py.

Team: The Outliers
Authors: Team The Outliers
"""
import sys
import os

repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)

from src.generate_3d_replay_html import main

if __name__ == "__main__":
    main()
