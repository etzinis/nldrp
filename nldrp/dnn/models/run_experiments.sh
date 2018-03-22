#!/bin/sh
python experiment.py --task cross --features fused
python experiment.py --task cross --features linear
python experiment.py --task independent --features fused
python experiment.py --task independent --features linear
python experiment.py --task dependent --features fused
python experiment.py --task dependent --features linear
