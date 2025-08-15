#!/usr/bin/env python
import os, sys
sys.path.append("src")
from geo_tracker.runner import run_panel
from geo_tracker.config import SETTINGS
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="default")
    parser.add_argument("--market", default=SETTINGS.market_default)
    parser.add_argument("--language", default=SETTINGS.language_default)
    parser.add_argument("--comments", default=None)
    args = parser.parse_args()
    run_panel(args.panel, args.market, args.language, args.comments)
