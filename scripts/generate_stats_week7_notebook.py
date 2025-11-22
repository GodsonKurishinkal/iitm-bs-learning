#!/usr/bin/env python3
"""Generate Week 7: Probability Basics"""
import json

def cm(c): return {"cell_type": "markdown", "metadata": {}, "source": c.split('\n')}
def cc(c): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.split('\n')}

nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.9.6"}}, "nbformat": 4, "nbformat_minor": 4}

nb["cells"].extend([
cm("""# Week 7: Probability Basics

**Course**: BSMA1002 - Statistics for Data Science I  
**Topic**: Foundations of Probability Theory

## Learning Objectives
- Understand probability fundamentals and axioms
- Calculate probabilities using counting methods
- Apply conditional probability and Bayes' theorem
- Understand independence and mutual exclusivity
- Solve probability problems with visualizations
- Apply probability to real-world scenarios"""),

cc("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import factorial, comb, perm
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
print("✓ Libraries loaded successfully")"""),

cm("""## 1. What is Probability?

**Probability**: Measure of likelihood that an event will occur

$$P(A) = \\frac{\\text{Number of favorable outcomes}}{\\text{Total number of possible outcomes}}$$

### Key Properties (Axioms)
1. $0 \\leq P(A) \\leq 1$ for any event A
2. $P(\\text{certain event}) = 1$
3. $P(A \\cup B) = P(A) + P(B)$ if A and B are mutually exclusive

### Terminology
- **Experiment**: Process with uncertain outcome
- **Sample Space (S)**: All possible outcomes
- **Event**: Subset of sample space
- **Complement**: $P(A') = 1 - P(A)$"""),

cc("""# Basic probability examples
print("Basic Probability Examples")
print("="*70)

# Fair die
sample_space = list(range(1, 7))
event_even = [2, 4, 6]
prob_even = len(event_even) / len(sample_space)
print(f"Die roll - P(even) = {len(event_even)}/{len(sample_space)} = {prob_even:.4f}")

# Deck of cards
total_cards = 52
hearts = 13
prob_heart = hearts / total_cards
print(f"Card draw - P(Heart) = {hearts}/{total_cards} = {prob_heart:.4f}")

# Complement
prob_rain = 0.3
prob_no_rain = 1 - prob_rain
print(f"Weather - P(No Rain) = 1 - {prob_rain} = {prob_no_rain:.1f}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

outcomes = sample_space
probs = [1/6] * 6
colors = ['green' if x in event_even else 'lightblue' for x in outcomes]
ax1.bar(outcomes, probs, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Outcome', fontsize=12)
ax1.set_ylabel('Probability', fontsize=12)
ax1.set_title(f'Fair Die: P(Even) = {prob_even:.4f}', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

categories = ['Rain', 'No Rain']
probs = [prob_rain, prob_no_rain]
colors = ['blue', 'yellow']
ax2.bar(categories, probs, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Probability', fontsize=12)
ax2.set_title('Complement Rule', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()"""),

cm("""## 2. Conditional Probability & Independence

**Conditional Probability**: Probability of A given B has occurred

$$P(A|B) = \\frac{P(A \\cap B)}{P(B)}$$

**Independence**: Events A and B are independent if:
$$P(A|B) = P(A)$$ or $$P(A \\cap B) = P(A) \\times P(B)$$"""),

cc("""# Conditional probability
print("Conditional Probability Examples")
print("="*70)

# Cards without replacement
print("\\nDrawing cards WITHOUT replacement:")
p_heart_1 = 13/52
p_heart_2_given_heart_1 = 12/51
p_both_hearts = p_heart_1 * p_heart_2_given_heart_1
print(f"P(1st Heart) = {p_heart_1:.4f}")
print(f"P(2nd Heart | 1st Heart) = {p_heart_2_given_heart_1:.4f}")
print(f"P(Both Hearts) = {p_both_hearts:.4f}")

# Medical testing
print("\\nMedical Test:")
p_disease = 0.01
p_pos_given_disease = 0.95
p_pos_given_healthy = 0.05

p_disease_and_pos = p_disease * p_pos_given_disease
p_healthy_and_pos = (1 - p_disease) * p_pos_given_healthy
p_positive = p_disease_and_pos + p_healthy_and_pos

print(f"P(Disease) = {p_disease:.2f}")
print(f"P(Positive | Disease) = {p_pos_given_disease:.2f}")
print(f"P(Positive) = {p_positive:.4f}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Card probabilities
scenarios = ['With\\nReplacement', 'Without\\nReplacement']
p_with = p_heart_1 * p_heart_1
p_without = p_both_hearts
probs = [p_with, p_without]
ax1.bar(scenarios, probs, color='pink', edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Probability', fontsize=12)
ax1.set_title('P(Both Hearts)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Test results
categories = ['Disease\\n& Pos', 'Healthy\\n& Pos']
values = [p_disease_and_pos, p_healthy_and_pos]
colors = ['red', 'orange']
ax2.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Probability', fontsize=12)
ax2.set_title('Positive Test Breakdown', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()"""),

cm("""## 3. Bayes' Theorem

**Update probabilities with new evidence**:

$$P(A|B) = \\frac{P(B|A) \\times P(A)}{P(B)}$$

Components:
- $P(A)$: Prior probability
- $P(A|B)$: Posterior probability
- $P(B|A)$: Likelihood"""),

cc("""# Bayes' theorem
print("Bayes' Theorem Application")
print("="*70)

# Spam detection
print("\\nSpam Email Detection:")
p_spam = 0.2
p_word_given_spam = 0.8
p_word_given_legit = 0.1

p_legit = 1 - p_spam
p_keyword = (p_word_given_spam * p_spam) + (p_word_given_legit * p_legit)
p_spam_given_word = (p_word_given_spam * p_spam) / p_keyword

print(f"Prior: P(Spam) = {p_spam:.2f}")
print(f"Posterior: P(Spam | Keyword) = {p_spam_given_word:.4f}")
print(f"Evidence updated belief from {p_spam:.0%} to {p_spam_given_word:.0%}")

# Medical diagnosis
p_disease_prior = 0.01
p_pos_given_disease = 0.95
p_pos_given_healthy = 0.05

p_positive = (p_pos_given_disease * p_disease_prior) + (p_pos_given_healthy * (1-p_disease_prior))
p_disease_given_pos = (p_pos_given_disease * p_disease_prior) / p_positive

print(f"\\nMedical Test:")
print(f"Before test: P(Disease) = {p_disease_prior:.1%}")
print(f"After positive: P(Disease | +) = {p_disease_given_pos:.1%}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

categories = ['Prior', 'Posterior']
probs = [p_spam, p_spam_given_word]
colors = ['lightblue', 'red']
bars = ax1.bar(categories, probs, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('P(Spam)', fontsize=12)
ax1.set_title('Spam Detection: Updating Beliefs', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
for bar, prob in zip(bars, probs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{prob:.2%}', ha='center', fontsize=11, fontweight='bold')

categories = ['Prior', 'Posterior']
probs = [p_disease_prior, p_disease_given_pos]
colors = ['lightgreen', 'orange']
bars = ax2.bar(categories, probs, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('P(Disease)', fontsize=12)
ax2.set_title('Medical Test: Updating Beliefs', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
for bar, prob in zip(bars, probs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{prob:.1%}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()"""),

cm("""## 4. Real Application: A/B Testing

Use probability to make data-driven decisions."""),

cc("""# A/B testing simulation
np.random.seed(42)

print("A/B Testing: Website Conversion")
print("="*70)

# Simulate
n_users_a, n_users_b = 1000, 1000
true_rate_a, true_rate_b = 0.10, 0.12

conversions_a = np.random.binomial(1, true_rate_a, n_users_a)
conversions_b = np.random.binomial(1, true_rate_b, n_users_b)

obs_rate_a = conversions_a.mean()
obs_rate_b = conversions_b.mean()
improvement = (obs_rate_b - obs_rate_a) / obs_rate_a

print(f"Group A: {conversions_a.sum()}/{n_users_a} = {obs_rate_a:.2%}")
print(f"Group B: {conversions_b.sum()}/{n_users_b} = {obs_rate_b:.2%}")
print(f"Improvement: {improvement:+.1%}")

# Significance
pooled = (conversions_a.sum() + conversions_b.sum()) / (n_users_a + n_users_b)
se = np.sqrt(pooled * (1 - pooled) * (1/n_users_a + 1/n_users_b))
z_score = (obs_rate_b - obs_rate_a) / se
significant = abs(z_score) > 1.96

print(f"Z-score: {z_score:.2f}")
print(f"Significant (95%): {significant}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

groups = ['Control\\n(A)', 'Treatment\\n(B)']
rates = [obs_rate_a, obs_rate_b]
colors = ['lightblue', 'lightgreen']
bars = ax1.bar(groups, rates, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Conversion Rate', fontsize=12)
ax1.set_title('Observed Conversion Rates', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
for bar, rate in zip(bars, rates):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{rate:.2%}', ha='center', fontsize=11, fontweight='bold')

categories = ['Converted\\n(A)', 'Not\\n(A)', 'Converted\\n(B)', 'Not\\n(B)']
values = [conversions_a.sum(), n_users_a - conversions_a.sum(),
          conversions_b.sum(), n_users_b - conversions_b.sum()]
colors = ['green', 'lightgray', 'darkgreen', 'lightgray']
ax2.bar(range(4), values, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_xticks(range(4))
ax2.set_xticklabels(categories, fontsize=9)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('Conversion Breakdown', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print(f"\\n{'✅ Deploy B' if significant else '⚠️ Need more data'}")"""),

cm("""## Summary

### Key Concepts

| Concept | Formula | Use Case |
|---------|---------|----------|
| **Basic** | $P(A) = n(A)/n(S)$ | Equally likely outcomes |
| **Complement** | $P(A') = 1 - P(A)$ | Opposite event |
| **Conditional** | $P(A|B) = P(A \\cap B)/P(B)$ | Given information |
| **Independence** | $P(A \\cap B) = P(A)P(B)$ | Events don't affect each other |
| **Bayes** | $P(A|B) = P(B|A)P(A)/P(B)$ | Update with evidence |

### Applications
- Machine learning classifiers
- A/B testing decisions
- Risk assessment
- Medical diagnosis
- Spam detection

---
**Next Week**: Random Variables and Distributions""")
])

output = "/Users/godsonkurishinkal/Projects/iitm-bs-learning/01-Foundation-Level/02-Statistics-I/notebooks/week-07-probability-basics.ipynb"
with open(output, 'w') as f: json.dump(nb, f, indent=2)
print(f"✓ Week 7: {len(nb['cells'])} cells")
