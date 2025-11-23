# Week 7: Introduction to Probability - Sample Spaces, Events, and Basic Rules

---
Date: 2025-11-22
Course: BSMA1002 - Statistics for Data Science I
Level: Foundation
Week: 7 of 12
Source: IIT Madras Statistics I Week 7
Topic Area: Probability Theory - Foundations
Tags: #BSMA1002 #Probability #SampleSpace #Events #ProbabilityRules #Week7 #Foundation
---

## Overview (BLUF)

**One-sentence definition**: Probability quantifies **uncertainty** by assigning numerical values (0 to 1) to outcomes of random experiments, enabling us to reason about chance events mathematically.

**Why it matters**: Probability is the mathematical foundation of statistics, machine learning, and AI. Every statistical test, confidence interval, prediction interval, and ML algorithm fundamentally relies on probability. Without it, we can only describe what happened (descriptive statistics), not what might happen (inferential statistics) or how confident we are in predictions.

**When to use**: Whenever outcomes are uncertain—dice rolls, stock prices, customer behavior, medical diagnoses, A/B test results, model predictions. Probability lets us quantify "how likely?" rather than just saying "maybe" or "probably."

**Real-world impact**:
- **ML**: Loss functions, likelihood estimation, Bayesian inference
- **A/B testing**: Statistical significance calculations
- **Risk assessment**: Insurance, finance, safety engineering
- **Natural language**: "What's the probability this email is spam?"

**Prerequisites**: Counting principles ([week-05](week-05-dispersion-variability.md)), combinations ([week-06](week-06-correlation-association.md)), set theory.

---

## Core Theory

### 1. Fundamental Concepts

#### 1.1 Random Experiment

**Definition**: Process whose outcome cannot be predicted with certainty, but we know all possible outcomes.

**Examples**:
- Flip a coin → {H, T}
- Roll a die → {1, 2, 3, 4, 5, 6}
- Measure height of random person → [0, ∞)
- Number of website clicks tomorrow → {0, 1, 2, ...}

**Key property**: Repeatability (can run experiment multiple times under same conditions).

#### 1.2 Sample Space (S or Ω)

**Definition**: Set of **all possible outcomes** of random experiment.

**Notation**: S, Ω (omega), U (universal set)

**Examples**:

| Experiment | Sample Space |
|------------|--------------|
| Flip coin | S = {H, T} |
| Roll die | S = {1, 2, 3, 4, 5, 6} |
| Flip coin twice | S = {HH, HT, TH, TT} |
| Count customers (0-100) | S = {0, 1, 2, ..., 100} |
| Measure height (cm) | S = (0, ∞) or [0, 300] |

**Types**:
- **Finite**: Countable, fixed number (die roll)
- **Countably infinite**: {0, 1, 2, ...} (count website visits forever)
- **Uncountably infinite**: Continuous range (height, weight, time)

#### 1.3 Event

**Definition**: **Subset** of sample space. Collection of outcomes.

**Notation**: Capital letters A, B, C, E

**Examples** (die roll, S = {1,2,3,4,5,6}):
- Event A = "even number" = {2, 4, 6}
- Event B = "number > 4" = {5, 6}
- Event C = "prime number" = {2, 3, 5}

**Types**:
- **Simple event**: Single outcome (A = {3})
- **Compound event**: Multiple outcomes (A = {2,4,6})
- **Certain event**: S itself (always occurs)
- **Impossible event**: ∅ empty set (never occurs)

#### Example 1: Sample Space for Two Coins

```python
import itertools

# Outcomes for single coin
coin_outcomes = ['H', 'T']

# Two-coin experiment
two_coins = list(itertools.product(coin_outcomes, repeat=2))
print(f"Sample space (2 coins): {two_coins}")
print(f"|S| = {len(two_coins)}")

# Define events
event_both_heads = [outcome for outcome in two_coins if outcome == ('H', 'H')]
event_exactly_one_head = [outcome for outcome in two_coins if outcome.count('H') == 1]
event_at_least_one_head = [outcome for outcome in two_coins if 'H' in outcome]

print(f"\nEvent 'Both heads': {event_both_heads}")
print(f"Event 'Exactly one head': {event_exactly_one_head}")
print(f"Event 'At least one head': {event_at_least_one_head}")
```

---

### 2. Set Operations on Events

Events are sets, so we use set operations:

#### 2.1 Complement (A' or $A^c$)

**Definition**: All outcomes **not** in A.

**Formula**: $A' = S - A$

**Example** (die roll):
- A = {2, 4, 6} (even)
- A' = {1, 3, 5} (odd)

#### 2.2 Union (A ∪ B)

**Definition**: Outcomes in A **or** B (or both).

**Example**:
- A = {2, 4, 6} (even)
- B = {5, 6} (>4)
- A ∪ B = {2, 4, 5, 6}

#### 2.3 Intersection (A ∩ B)

**Definition**: Outcomes in **both** A and B.

**Example**:
- A = {2, 4, 6}
- B = {5, 6}
- A ∩ B = {6}

#### 2.4 Mutually Exclusive (Disjoint)

**Definition**: A and B have **no outcomes in common**.

**Condition**: A ∩ B = ∅

**Example**:
- A = {2, 4, 6} (even)
- C = {1, 3, 5} (odd)
- A ∩ C = ∅ → mutually exclusive

#### Example 2: Venn Diagrams

```python
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles

# Example: Die roll
S = {1, 2, 3, 4, 5, 6}
A = {2, 4, 6}  # Even
B = {5, 6}     # > 4

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Complement
v1 = venn2(subsets=(len(A), 0, 0), set_labels=('A (even)', ''))
v1.get_label_by_id('10').set_text(f'A = {A}')
axes[0].set_title("Complement A'", fontsize=14, fontweight='bold')
axes[0].text(0.5, -0.3, f"A' = {S - A}", ha='center', fontsize=12, transform=axes[0].transAxes)

# Union
v2 = venn2(subsets=(len(A - B), len(B - A), len(A & B)), set_labels=('A', 'B'))
axes[1].set_title("Union A ∪ B", fontsize=14, fontweight='bold')
axes[1].text(0.5, -0.3, f"A ∪ B = {A | B}", ha='center', fontsize=12, transform=axes[1].transAxes)

# Intersection
v3 = venn2(subsets=(len(A - B), len(B - A), len(A & B)), set_labels=('A', 'B'))
# Highlight intersection
patch = v3.get_patch_by_id('11')
if patch:
    patch.set_color('red')
    patch.set_alpha(0.7)
axes[2].set_title("Intersection A ∩ B", fontsize=14, fontweight='bold')
axes[2].text(0.5, -0.3, f"A ∩ B = {A & B}", ha='center', fontsize=12, transform=axes[2].transAxes)

plt.tight_layout()
plt.show()
```

---

### 3. Definitions of Probability

#### 3.1 Classical (Theoretical) Probability

**Definition**: When all outcomes are **equally likely**:
$$P(A) = \frac{\text{Number of outcomes in A}}{\text{Total number of outcomes in S}} = \frac{|A|}{|S|}$$

**When valid**: Symmetry (fair coin, unbiased die, well-shuffled cards)

**Example**: Fair die, A = {2, 4, 6}
$$P(A) = \frac{3}{6} = \frac{1}{2}$$

#### 3.2 Empirical (Frequentist) Probability

**Definition**: Based on **observed data**:
$$P(A) \approx \frac{\text{Number of times A occurred}}{\text{Total number of trials}}$$

**When used**: When outcomes not equally likely or unknown.

**Example**: Flip coin 1000 times, get 527 heads
$$P(H) \approx \frac{527}{1000} = 0.527$$

#### 3.3 Subjective Probability

**Definition**: Based on **belief/expert judgment** (no data or symmetry).

**Examples**:
- "Probability that AI will surpass human intelligence by 2050"
- "Probability our startup succeeds"

#### Example 3: Comparing Definitions

```python
import numpy as np
import random

# Classical: Fair die
die_faces = [1, 2, 3, 4, 5, 6]
classical_prob_even = len([x for x in die_faces if x % 2 == 0]) / len(die_faces)
print(f"Classical P(even) = {classical_prob_even}")

# Empirical: Simulate die rolls
np.random.seed(42)
n_rolls = 10000
rolls = np.random.randint(1, 7, n_rolls)
empirical_prob_even = np.sum(rolls % 2 == 0) / n_rolls
print(f"Empirical P(even) after {n_rolls} rolls = {empirical_prob_even:.4f}")

# Convergence to classical
trial_sizes = [10, 100, 1000, 10000, 100000]
empirical_probs = []

for n in trial_sizes:
    rolls_n = np.random.randint(1, 7, n)
    prob_n = np.sum(rolls_n % 2 == 0) / n
    empirical_probs.append(prob_n)
    print(f"  n={n:6d}: P(even) = {prob_n:.4f}")

# Visualize convergence
plt.figure(figsize=(12, 6))
plt.plot(trial_sizes, empirical_probs, 'o-', linewidth=2, markersize=10, label='Empirical')
plt.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Theoretical (0.5)')
plt.xscale('log')
plt.xlabel('Number of Trials', fontsize=12)
plt.ylabel('P(Even)', fontsize=12)
plt.title('Law of Large Numbers: Empirical → Theoretical', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

---

### 4. Axioms of Probability (Kolmogorov)

These are the **fundamental rules** all probabilities must satisfy:

**Axiom 1** (Non-negativity):
$$P(A) \geq 0 \quad \text{for any event A}$$

**Axiom 2** (Certainty):
$$P(S) = 1 \quad \text{(something must happen)}$$

**Axiom 3** (Additivity): If A and B are mutually exclusive (disjoint):
$$P(A \cup B) = P(A) + P(B)$$

**All other probability rules derive from these three axioms!**

---

### 5. Basic Probability Rules

#### 5.1 Complement Rule

$$P(A') = 1 - P(A)$$

**Proof**:
- A and A' are mutually exclusive
- A ∪ A' = S
- Therefore: P(A) + P(A') = P(S) = 1
- Rearrange: P(A') = 1 - P(A) ✓

**Application**: Often easier to compute P(A') than P(A).

**Example**: P(at least one head in 10 flips)?
- Direct: P(H on flip 1 OR flip 2 OR ... OR flip 10) → complicated
- Complement: P(at least one H) = 1 - P(no heads) = 1 - (1/2)^10 = 0.999

#### 5.2 Addition Rule (General)

For **any** two events A and B:
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

**Why subtract?** When adding P(A) + P(B), we double-count the overlap A ∩ B.

**Special case**: If A and B mutually exclusive (A ∩ B = ∅):
$$P(A \cup B) = P(A) + P(B)$$

#### Example 4: Addition Rule

**Problem**: Card deck (52 cards). Draw one card. What's P(heart OR face card)?

**Solution**:
- A = heart (13 cards)
- B = face card (12 cards: J, Q, K in 4 suits)
- A ∩ B = heart face cards (3 cards: J♥, Q♥, K♥)

$$P(A \cup B) = P(A) + P(B) - P(A \cap B) = \frac{13}{52} + \frac{12}{52} - \frac{3}{52} = \frac{22}{52} = \frac{11}{26}$$

```python
# Verify by enumeration
deck = []
for suit in ['♠', '♥', '♦', '♣']:
    for rank in ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']:
        deck.append(rank + suit)

hearts = [card for card in deck if '♥' in card]
face_cards = [card for card in deck if card[0] in ['J', 'Q', 'K']]
heart_or_face = [card for card in deck if '♥' in card or card[0] in ['J', 'Q', 'K']]
heart_and_face = [card for card in deck if '♥' in card and card[0] in ['J', 'Q', 'K']]

print(f"P(heart) = {len(hearts)}/52 = {len(hearts)/52:.4f}")
print(f"P(face) = {len(face_cards)}/52 = {len(face_cards)/52:.4f}")
print(f"P(heart ∩ face) = {len(heart_and_face)}/52 = {len(heart_and_face)/52:.4f}")
print(f"P(heart ∪ face) = {len(heart_or_face)}/52 = {len(heart_or_face)/52:.4f}")

# Verify addition rule
addition_rule = len(hearts)/52 + len(face_cards)/52 - len(heart_and_face)/52
print(f"\nAddition rule: {len(hearts)/52:.4f} + {len(face_cards)/52:.4f} - {len(heart_and_face)/52:.4f} = {addition_rule:.4f}")
```

#### 5.3 Properties of Probability

1. **Range**: $0 \leq P(A) \leq 1$ for any event A

2. **Impossible event**: $P(\emptyset) = 0$

3. **Certain event**: $P(S) = 1$

4. **Monotonicity**: If $A \subseteq B$, then $P(A) \leq P(B)$

5. **Difference**: $P(B - A) = P(B) - P(A \cap B)$

6. **If A ⊆ B**: $P(B - A) = P(B) - P(A)$

---

## Data Science Applications

### 1. A/B Testing - Baseline Probabilities

**Problem**: E-commerce site. Historical conversion rate 10%. Run A/B test.

```python
# Historical data
n_visitors = 10000
n_conversions = 1000
baseline_prob = n_conversions / n_visitors

print(f"Baseline conversion probability: {baseline_prob:.2%}")

# Simulate conversions
np.random.seed(42)
conversions = np.random.binomial(1, baseline_prob, n_visitors)
empirical_prob = conversions.sum() / n_visitors

print(f"Simulated conversion rate: {empirical_prob:.2%}")

# Confidence: How likely to get ≤ 950 conversions if true rate is 10%?
# (We'll cover this in hypothesis testing, but using complement rule)
from scipy import stats
prob_extreme = stats.binom.cdf(950, n_visitors, baseline_prob)
print(f"\nP(≤950 conversions | p=0.10) = {prob_extreme:.4f}")
print(f"P(>950 conversions | p=0.10) = {1 - prob_extreme:.4f} (complement rule)")
```

### 2. Spam Filter - Event Probabilities

**Problem**: Estimate P(spam) from email dataset.

```python
# Email dataset
emails = {
    'spam': 450,
    'ham': 1550
}

total_emails = sum(emails.values())
prob_spam = emails['spam'] / total_emails
prob_ham = emails['ham'] / total_emails

print(f"P(spam) = {prob_spam:.3f}")
print(f"P(ham) = {prob_ham:.3f}")
print(f"P(spam) + P(ham) = {prob_spam + prob_ham:.3f} (should be 1.0)")

# Complement rule
print(f"\nUsing complement: P(ham) = 1 - P(spam) = {1 - prob_spam:.3f}")
```

### 3. Model Evaluation - Classification Probabilities

**Problem**: Binary classifier outputs probabilities. Evaluate calibration.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Get predicted probabilities
probs = model.predict_proba(X_test)[:, 1]

# Empirical probability
empirical_prob_positive = y_test.sum() / len(y_test)
print(f"Empirical P(class=1) = {empirical_prob_positive:.3f}")

# Calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, probs, n_bins=10)

plt.figure(figsize=(10, 6))
plt.plot(mean_predicted_value, fraction_of_positives, 'o-', linewidth=2, markersize=8)
plt.plot([0, 1], [0, 1], 'r--', label='Perfectly calibrated')
plt.xlabel('Predicted Probability', fontsize=12)
plt.ylabel('Empirical Probability', fontsize=12)
plt.title('Calibration Curve', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Common Pitfalls and Misconceptions

### Pitfall 1: Confusing P(A) with P(A|B)

❌ **Wrong**: "P(rain today) = 30%, so P(rain | cloudy) = 30%"

✅ **Right**: Conditional probability P(rain | cloudy) may be much higher (we'll cover Week 8)

### Pitfall 2: Adding Non-Exclusive Event Probabilities

❌ **Wrong**: "P(A) = 0.6, P(B) = 0.5 → P(A ∪ B) = 1.1"

✅ **Right**: Must subtract overlap: P(A ∪ B) = 0.6 + 0.5 - P(A ∩ B)

### Pitfall 3: Assuming Zero Probability = Impossible

❌ **Wrong**: "Height exactly 170.0000... cm has P = 0, so impossible"

✅ **Right**: For continuous variables, P(exact value) = 0, but event is possible (use intervals)

### Pitfall 4: Gambler's Fallacy

❌ **Wrong**: "Coin flipped 5 heads in row, so next flip more likely tails"

✅ **Right**: Each flip independent, P(H) = P(T) = 0.5 regardless of history

### Pitfall 5: Misinterpreting Complement

❌ **Wrong**: "P(A) = 0.3, so P(not exactly A) = 0.7"

✅ **Right**: P(A') = 0.7 is correct, but "not exactly A" may be ambiguous in context

---

## Self-Assessment and Active Recall

### Concept Check Questions

1. **Define**: Sample space, event, mutually exclusive events.

2. **Distinguish**: Classical vs empirical vs subjective probability.

3. **Apply**: Why is P(A) + P(B) - P(A ∩ B) needed for union?

4. **Calculate**: If P(A) = 0.4, what is P(A')?

5. **Verify**: Show that probabilities in sample space S = {1,2,3,4,5,6} (fair die) satisfy axioms.

### Practice Problems

#### Basic Level

1. Fair coin flipped twice. List sample space. What's P(at least one head)?

2. Die rolled. A = {1,3,5}, B = {4,5,6}. Find:
   a) P(A)
   b) P(A ∪ B)
   c) P(A ∩ B)

3. Deck of 52 cards. P(drawing ace)?

#### Intermediate Level

4. Two dice rolled. What's P(sum = 7)?

5. If P(A) = 0.6, P(B) = 0.4, P(A ∩ B) = 0.2, find:
   a) P(A ∪ B)
   b) P(A')
   c) P(A ∪ B)'

6. Simulate 10,000 coin flips. How close is empirical P(H) to 0.5?

#### Advanced Level

7. Prove: If A ⊆ B, then P(A) ≤ P(B).

8. Prove: P(A ∪ B ∪ C) = P(A) + P(B) + P(C) - P(A∩B) - P(A∩C) - P(B∩C) + P(A∩B∩C)

9. Three events A, B, C are mutually exclusive. P(A)=0.2, P(B)=0.3, P(C)=0.4. What's P(A ∪ B ∪ C)?

---

## Quick Reference Summary

### Key Formulas

**Complement**:
$$P(A') = 1 - P(A)$$

**Addition Rule (General)**:
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

**Addition Rule (Mutually Exclusive)**:
$$P(A \cup B) = P(A) + P(B) \quad \text{if } A \cap B = \emptyset$$

**Classical Probability**:
$$P(A) = \frac{|A|}{|S|}$$

### Three Axioms of Probability

1. $P(A) \geq 0$
2. $P(S) = 1$
3. $P(A \cup B) = P(A) + P(B)$ if A ∩ B = ∅

### Python Templates

```python
import numpy as np
import random

# Classical probability (equally likely)
prob_A = len(event_A) / len(sample_space)

# Empirical probability (simulation)
def simulate_experiment(n_trials):
    successes = sum([trial() for _ in range(n_trials)])
    return successes / n_trials

# Complement
prob_not_A = 1 - prob_A

# Union (general)
prob_union = prob_A + prob_B - prob_intersection

# Check axioms
assert 0 <= prob_A <= 1
assert prob_A + prob_not_A == 1.0
```

### Top 3 Things to Remember

1. **All probabilities: 0 ≤ P(A) ≤ 1**, and **P(S) = 1**
2. **Complement rule**: P(A') = 1 - P(A) (often easier to compute)
3. **Addition rule**: Subtract P(A ∩ B) to avoid double-counting

---

## Further Resources

### Documentation
- Python `random` module
- NumPy random generation
- SciPy `stats` distributions

### Books
- Sheldon Ross, "A First Course in Probability" - Chapters 1-2
- Feller, "An Introduction to Probability Theory" - Chapter 1

### Interactive
- Seeing Theory (https://seeing-theory.brown.edu/) - Visual probability
- GeoGebra probability simulations

### Review Schedule
- **After 1 day**: Work through axioms and basic rules by hand
- **After 3 days**: Simulate experiments (coins, dice, cards)
- **After 1 week**: Apply to A/B testing probabilities
- **After 2 weeks**: Connect to conditional probability (Week 8)

---

**Related Notes**:
- Previous: [week-06-correlation-association.md](week-06-correlation-association.md)
- Next: [week-08-random-variables.md](week-08-random-variables.md)
- Foundation: Uses counting from Weeks 5-6

**Last Updated**: 2025-11-22
**Status**: Complete and comprehensive ✅
