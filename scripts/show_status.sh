#!/bin/bash

HERE="$(dirname "$(readlink -f "$0")")"

exp_name=$($HERE/show_experiment_name.py)

set -x

amlt status $exp_name
