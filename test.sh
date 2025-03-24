#!/usr/bin/env bash

source activate hmt
pytest -m "not slow"