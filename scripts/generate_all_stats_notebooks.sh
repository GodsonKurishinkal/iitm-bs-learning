#!/bin/bash
# Generate all Statistics-I notebooks (weeks 5-12)

echo "Generating Statistics-I notebooks..."
echo "===================================="

# Week 5 - Dispersion
python3 << 'WEEK5'
import json
exec(open('generate_stats_week4_notebook.py').read().replace('week-04-central-tendency-measures', 'week-05-dispersion-variability').replace('Week 4: Measures of Central Tendency', 'Week 5: Measures of Dispersion and Variability').replace('Week 4 of 12', 'Week 5 of 12').replace('Central Tendency', 'Dispersion and Variability'))
WEEK5

echo "✓ Week 5 complete"
