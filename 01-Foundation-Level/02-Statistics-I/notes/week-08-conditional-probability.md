# Week 8: Conditional Probability, Independence, and Bayes' Theorem

---
Date: 2025-11-22
Course: BSMA1002 - Statistics for Data Science I
Level: Foundation
Week: 8 of 12
Source: IIT Madras Statistics I Week 8
Topic Area: Probability Theory - Conditional Probability
Tags: #BSMA1002 #ConditionalProbability #Independence #BayesTheorem #Week8 #Foundation
---

## Overview (BLUF)

**One-sentence definition**: Conditional probability P(A|B) updates probability of A based on new information B—the foundation of Bayesian reasoning, medical diagnostics, spam filters, and modern ML.

**Why it matters**: Real-world probabilities are conditional. Bayes' theorem reverses conditionals (know P(test|disease), want P(disease|test)), powering recommendation systems, autonomous vehicles, and AI decision-making.

**When to use**: Medical testing, A/B testing, spam filtering, feature importance—any scenario where new information updates beliefs.

**Prerequisites**: Basic probability ([week-07](week-07-probability-basics.md)).

---

## Core Theory

### 1. Conditional Probability

**Definition**:
$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

"Of the times B occurs, what fraction also has A?"

#### Example: Die Roll  
Roll die. Given even, what's P(>4)?
- B = {2,4,6}, A = {5,6}, A∩B = {6}
- P(A|B) = 1/3

---

### 2. Independence

**Definition**: A and B independent if:
$$P(A \cap B) = P(A) \cdot P(B)$$

Equivalent: P(A|B) = P(A) (knowing B doesn't change A)

---

### 3. Bayes' Theorem

$$P(A|B) = \frac{P(B|A) P(A)}{P(B)}$$

**Terms**: P(A)=prior, P(B|A)=likelihood, P(A|B)=posterior

#### Medical Test Example
- Disease: 1% prevalence  
- Test: 95% sensitive, 5% false positive
- P(disease|+) = 16.1% only!

---

## Quick Reference

**Conditional**: P(A|B) = P(A∩B)/P(B)  
**Independence**: P(A∩B) = P(A)P(B)  
**Bayes**: P(A|B) = P(B|A)P(A)/P(B)

**Last Updated**: 2025-11-22 ✅
