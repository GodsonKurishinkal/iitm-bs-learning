# Week 1: Introduction to Statistics and Data Types

**Date**: 2025-11-16  
**Course**: Statistics for Data Science I (BSMA1002)

## Topics Covered

1. Introduction to Statistics
2. Types of Data (Qualitative vs Quantitative)
3. Descriptive vs Inferential Statistics
4. Scales of Measurement (Nominal, Ordinal, Interval, Ratio)

---

## Key Concepts

### 1. What is Statistics?

**Statistics** is the science of collecting, organizing, analyzing, interpreting, and presenting data.

#### Two Main Branches

**Descriptive Statistics**
- Summarizes and describes data features
- Uses measures like mean, median, mode, standard deviation
- Visual tools: histograms, bar charts, box plots
- **Example**: Average age of students in a class = 22 years
- **Data Science use**: Exploratory Data Analysis (EDA)

**Inferential Statistics**
- Makes predictions and inferences about populations from samples
- Uses hypothesis testing, confidence intervals, regression
- **Example**: Predicting election results from poll of 1,000 voters
- **Data Science use**: A/B testing, predictive modeling, machine learning

#### Why Statistics Matters in Data Science

1. **Understanding Data**: Identify patterns, trends, anomalies
2. **Making Decisions**: Evidence-based conclusions from data
3. **Building Models**: Foundation for ML algorithms
4. **Measuring Uncertainty**: Confidence in predictions
5. **Communicating Insights**: Present findings to stakeholders

---

### 2. Population vs Sample

**Population**
- The complete set of all items/individuals of interest
- Usually too large or impossible to study entirely
- Denoted by N (population size)
- **Example**: All smartphones manufactured globally

**Sample**
- A subset of the population selected for study
- Must be representative of the population
- Denoted by n (sample size)
- **Example**: 500 smartphones tested for quality

**Key Relationship**: We use **sample statistics** to estimate **population parameters**

| Concept | Population | Sample |
|---------|-----------|--------|
| Mean | μ (mu) | x̄ (x-bar) |
| Standard Deviation | σ (sigma) | s |
| Proportion | p | p̂ (p-hat) |
| Size | N | n |

---

### 3. Types of Data

Understanding data types is crucial for choosing the right statistical methods and visualizations.

#### A. Qualitative Data (Categorical)

Data that represents **categories** or **labels** (non-numeric).

**Nominal Data**
- Categories with no inherent order
- **Examples**: 
  - Gender: {Male, Female, Non-binary}
  - Blood type: {A, B, AB, O}
  - Programming language: {Python, R, Java, JavaScript}
  - City: {Mumbai, Delhi, Bangalore}
- **Operations**: Count frequencies, find mode
- **Visualization**: Bar chart, pie chart
- **DS Application**: Classification labels in ML

**Ordinal Data**
- Categories with a meaningful order/ranking
- **Examples**:
  - Education level: {High School < Bachelor's < Master's < PhD}
  - Satisfaction rating: {Very Dissatisfied < Dissatisfied < Neutral < Satisfied < Very Satisfied}
  - Movie rating: {1 star < 2 stars < 3 stars < 4 stars < 5 stars}
  - T-shirt size: {XS < S < M < L < XL}
- **Operations**: Count, mode, median, rank
- **Visualization**: Bar chart (ordered), cumulative frequency
- **DS Application**: Sentiment analysis, ranking systems

#### B. Quantitative Data (Numerical)

Data that represents **quantities** (numeric values).

**Discrete Data**
- Countable, finite or infinite
- Often integers (whole numbers)
- **Examples**:
  - Number of students: {0, 1, 2, 3, ...}
  - Number of defects: {0, 1, 2, ...}
  - Number of website clicks per day
  - Number of goals scored
- **Operations**: All mathematical operations
- **Visualization**: Bar chart, histogram, line plot
- **DS Application**: Count data, frequency models

**Continuous Data**
- Can take any value within a range (includes decimals)
- Measured, not counted
- **Examples**:
  - Height: 165.5 cm, 170.2 cm
  - Temperature: 25.7°C, 30.1°C
  - Time: 12.345 seconds
  - Income: ₹45,678.90
- **Operations**: All mathematical operations
- **Visualization**: Histogram, density plot, box plot, scatter plot
- **DS Application**: Regression, time series analysis

#### Quick Classification Guide

```
Is the data numeric?
├─ NO → Qualitative
│   └─ Is there a natural order?
│       ├─ YES → Ordinal
│       └─ NO → Nominal
└─ YES → Quantitative
    └─ Can it be counted?
        ├─ YES → Discrete
        └─ NO → Continuous
```

---

### 4. Scales of Measurement

The **scale of measurement** determines what statistical operations are valid.

#### Nominal Scale

- **Definition**: Categories without order
- **Properties**: 
  - Identity (each value unique)
  - No mathematical operations
- **Valid Operations**: 
  - Count (frequency)
  - Mode (most frequent)
  - Chi-square test
- **Invalid Operations**: Mean, median, comparison operators
- **Examples**: Gender, color, country, product category
- **Coding**: Can assign numbers (1=Male, 2=Female) but numbers are just labels

#### Ordinal Scale

- **Definition**: Categories with meaningful order
- **Properties**:
  - Identity
  - Order (ranking)
  - No equal intervals between ranks
- **Valid Operations**:
  - Count, mode, median
  - Percentiles, quartiles
  - Non-parametric tests (Mann-Whitney, Kruskal-Wallis)
- **Invalid Operations**: Mean, standard deviation (technically, but often approximated)
- **Examples**: Likert scales, education levels, rankings
- **Coding**: Numbers indicate order (1 < 2 < 3) but differences not equal

#### Interval Scale

- **Definition**: Numeric data with equal intervals, no true zero
- **Properties**:
  - Identity
  - Order
  - Equal intervals
  - No absolute zero (zero is arbitrary)
- **Valid Operations**:
  - Count, mode, median, mean
  - Standard deviation, variance
  - Addition, subtraction
  - Parametric tests (t-test, ANOVA)
- **Invalid Operations**: Multiplication, division, ratios
- **Examples**:
  - Temperature in Celsius/Fahrenheit (0° doesn't mean "no temperature")
  - Years (2000 is not "twice" 1000)
  - IQ scores
  - pH levels
- **Why no ratios**: 20°C is NOT twice as hot as 10°C

#### Ratio Scale

- **Definition**: Numeric data with equal intervals AND true zero
- **Properties**:
  - Identity
  - Order
  - Equal intervals
  - Absolute zero (zero means "none")
- **Valid Operations**:
  - ALL mathematical operations
  - Count, mode, median, mean
  - Standard deviation, variance
  - Addition, subtraction, multiplication, division
  - Ratios and proportions
- **Invalid Operations**: None (all operations valid)
- **Examples**:
  - Height (0 cm = no height)
  - Weight (0 kg = no weight)
  - Income (₹0 = no income)
  - Age (0 years = newborn)
  - Number of items
- **Why ratios work**: 20 kg IS twice 10 kg

#### Scale Comparison Table

| Feature | Nominal | Ordinal | Interval | Ratio |
|---------|---------|---------|----------|-------|
| Identity | ✓ | ✓ | ✓ | ✓ |
| Order | ✗ | ✓ | ✓ | ✓ |
| Equal Intervals | ✗ | ✗ | ✓ | ✓ |
| Absolute Zero | ✗ | ✗ | ✗ | ✓ |
| Mode | ✓ | ✓ | ✓ | ✓ |
| Median | ✗ | ✓ | ✓ | ✓ |
| Mean | ✗ | ✗ | ✓ | ✓ |
| Std Dev | ✗ | ✗ | ✓ | ✓ |
| Ratios | ✗ | ✗ | ✗ | ✓ |

---

## Definitions

- **Statistic**: A numerical measure computed from sample data (e.g., sample mean x̄)
- **Parameter**: A numerical measure that describes a population characteristic (e.g., population mean μ)
- **Variable**: A characteristic that varies among individuals (e.g., age, income, color)
- **Data**: Values that variables take (observations)
- **Census**: Study of the entire population
- **Survey**: Study of a sample from the population
- **Representative Sample**: A sample that accurately reflects the population characteristics
- **Bias**: Systematic error that makes sample unrepresentative

---

## Important Distinctions

### Data vs Information vs Knowledge

1. **Data**: Raw facts and figures (e.g., 23, 45, 67)
2. **Information**: Processed data with context (e.g., ages of three students)
3. **Knowledge**: Insights from information (e.g., average student age is 45)
4. **Wisdom**: Applied knowledge for decision-making

### Quantitative Methods

**Descriptive Statistics**
- Central tendency: Mean, median, mode
- Dispersion: Range, variance, standard deviation
- Position: Percentiles, quartiles

**Inferential Statistics**
- Estimation: Point estimates, confidence intervals
- Hypothesis testing: t-tests, ANOVA, chi-square
- Prediction: Regression, time series

---

## Examples (Worked Problems)

### Example 1: Identifying Data Types

**Problem**: Classify the following variables by data type:

a) Number of employees in a company  
b) Employee satisfaction (Low/Medium/High)  
c) Employee ID numbers  
d) Annual salary  
e) Temperature in office  
f) Department name  

**Solution**:

a) **Number of employees**: Quantitative - Discrete (counted)
b) **Employee satisfaction**: Qualitative - Ordinal (ordered categories)
c) **Employee ID**: Qualitative - Nominal (just labels, no mathematical meaning)
d) **Annual salary**: Quantitative - Continuous (can have decimals)
e) **Temperature**: Quantitative - Continuous (interval scale)
f) **Department name**: Qualitative - Nominal (no order)

### Example 2: Identifying Scales of Measurement

**Problem**: Identify the scale of measurement:

a) Movie ratings (1-5 stars)  
b) Height of students (in cm)  
c) Country of origin  
d) Semester GPA (0.0 - 10.0)  

**Solution**:

a) **Movie ratings**: Ordinal (ordered but unequal intervals; difference between 1-2 stars may not equal 4-5 stars)
b) **Height**: Ratio (equal intervals, true zero, can say someone is twice as tall)
c) **Country**: Nominal (categories, no order)
d) **GPA**: Interval (equal intervals, but 0 GPA doesn't mean "no knowledge"; can't say 8.0 is "twice" 4.0 in absolute terms)

### Example 3: Population vs Sample

**Problem**: A researcher wants to study smartphone usage patterns in India. They survey 2,000 smartphone users across 10 major cities.

a) What is the population?  
b) What is the sample?  
c) What is a potential bias?

**Solution**:

a) **Population**: All smartphone users in India (~750 million people)
b) **Sample**: The 2,000 users surveyed
c) **Potential bias**: 
   - Urban bias (only major cities, missing rural users)
   - Selection bias (voluntary participants may differ from general population)
   - Coverage bias (excludes those without smartphones)

### Example 4: Descriptive vs Inferential Statistics

**Problem**: Classify each statement:

a) "The average score on the test was 78%"  
b) "Based on the sample, we estimate that 60% of voters support the candidate"  
c) "The standard deviation of salaries in our company is ₹15,000"  
d) "Students who study more than 3 hours per day are likely to score above 85%"

**Solution**:

a) **Descriptive** (summarizing observed data)
b) **Inferential** (estimating population parameter from sample)
c) **Descriptive** (describing company data)
d) **Inferential** (making prediction based on pattern)

### Example 5: Valid Operations by Scale

**Problem**: For each scenario, determine if the operation is valid:

a) Finding the average gender (coded as 1=Male, 2=Female)  
b) Finding the median satisfaction level (1=Very Unsatisfied to 5=Very Satisfied)  
c) Calculating the ratio of two temperatures: 40°C / 20°C  
d) Calculating the ratio of two salaries: ₹80,000 / ₹40,000

**Solution**:

a) **Invalid** - Gender is nominal; average is meaningless
b) **Valid** - Satisfaction is ordinal; median makes sense for ordered data
c) **Invalid** - Celsius is interval scale (no true zero); ratios are meaningless
d) **Valid** - Salary is ratio scale; someone earning ₹80,000 does earn twice as much as someone earning ₹40,000

### Example 6: Real-World Data Classification

**Problem**: You're analyzing an e-commerce dataset. Classify these variables:

a) Customer age  
b) Product category (Electronics, Clothing, Books)  
c) Number of items in cart  
d) Customer review rating (1-5 stars)  
e) Order amount (₹)  
f) Delivery status (Pending, Shipped, Delivered)

**Solution**:

a) **Age**: Quantitative - Continuous (Ratio scale - true zero exists)
b) **Product category**: Qualitative - Nominal (no natural order)
c) **Items in cart**: Quantitative - Discrete (counted)
d) **Review rating**: Qualitative - Ordinal (ordered, but intervals not necessarily equal)
e) **Order amount**: Quantitative - Continuous (Ratio scale - ₹0 is true zero)
f) **Delivery status**: Qualitative - Ordinal (natural progression: Pending → Shipped → Delivered)

### Example 7: Data Science Application

**Problem**: A data scientist is building a house price prediction model with these features:

- Number of bedrooms
- Location (North, South, East, West)
- Square footage
- Age of house (years)
- Condition rating (Poor, Fair, Good, Excellent)

What encoding strategy should be used for each feature in the ML model?

**Solution**:

1. **Number of bedrooms**: Discrete quantitative → Use as-is (integer values)
2. **Location**: Nominal qualitative → One-hot encoding (create binary columns)
3. **Square footage**: Continuous quantitative → Normalize/standardize
4. **Age**: Ratio quantitative → Use as-is or normalize
5. **Condition rating**: Ordinal qualitative → Ordinal encoding (Poor=1, Fair=2, Good=3, Excellent=4)

**Reasoning**: The encoding preserves the inherent structure of each data type (order for ordinal, no order for nominal, numeric relationships for quantitative).

---

## Data Science Applications

### 1. Exploratory Data Analysis (EDA)

Understanding data types guides your EDA approach:

- **Nominal**: Frequency tables, bar charts, mode
- **Ordinal**: Frequency tables, ordered bar charts, median
- **Continuous**: Histograms, box plots, mean, standard deviation

### 2. Feature Engineering

Data type determines transformations:

- **Categorical**: One-hot encoding, label encoding, target encoding
- **Numerical**: Scaling, normalization, binning, polynomial features

### 3. Model Selection

Different models for different data:

- **Classification** (Qualitative target): Logistic regression, decision trees, SVM
- **Regression** (Quantitative target): Linear regression, polynomial regression, neural networks

### 4. Visualization Choice

| Data Type | Appropriate Visualizations |
|-----------|---------------------------|
| Nominal | Bar chart, pie chart, word cloud |
| Ordinal | Ordered bar chart, stacked bar chart |
| Discrete | Bar chart, histogram, stem plot |
| Continuous | Histogram, density plot, box plot, scatter plot |

---

## Practice Problems

### Basic Level

1. Classify as qualitative or quantitative: 
   a) ZIP code, b) Weight, c) Hair color, d) Number of siblings

2. Identify the scale: 
   a) Military rank, b) Distance, c) Eye color, d) Year

3. Is this descriptive or inferential?
   "A poll of 1,000 voters shows 52% support for the candidate. We conclude that the candidate will likely win."

### Intermediate Level

4. A dataset contains: Name, Age, Department, Years of Experience, Salary. 
   Identify the data type and scale for each variable.

5. Why can't you calculate the average of car colors in a parking lot? What scale is it?

6. You have satisfaction ratings (1-10). Someone argues "8 is twice as satisfied as 4". Is this correct? Why or why not?

### Advanced Level

7. Design a data collection form for a student survey. Include at least one variable of each data type (nominal, ordinal, discrete, continuous). Specify the scale for each.

8. Explain why temperature in Kelvin is ratio scale but Celsius is interval scale.

9. A company wants to analyze employee performance. They use:
   - Performance rating (Poor/Average/Good/Excellent)
   - Years at company
   - Salary
   - Department
   
   For each variable:
   a) Identify data type and scale
   b) Suggest appropriate descriptive statistics
   c) Suggest appropriate visualization

10. In machine learning, why is it important to correctly identify ordinal vs nominal data? Give an example where confusing them could lead to poor model performance.

---

## Questions/Doubts

- [ ] What's the difference between a parameter and a statistic?
- [ ] Why is distinguishing interval from ratio scale important in practice?
- [ ] How do I choose between treating ordinal data as categorical or numeric?
- [ ] What sampling methods ensure representative samples?

---

## Action Items

- [ ] Review all data types and scales - create flashcards
- [ ] Practice classifying variables from real datasets (Kaggle, UCI ML Repository)
- [ ] Work through Jupyter notebook examples with Python pandas
- [ ] Complete practice problems 1-10
- [ ] Watch lecture video on measurement scales

---

## Key Takeaways

1. **Statistics has two branches**: Descriptive (summarize) and Inferential (predict)
2. **Data types matter**: Qualitative vs Quantitative determines your analysis approach
3. **Scales are hierarchical**: Nominal < Ordinal < Interval < Ratio (each adds properties)
4. **Choose methods wisely**: Data type restricts which statistical operations are valid
5. **In Data Science**: Proper data type identification is crucial for feature engineering and model selection

---

## References

- Textbook: Introduction to Statistics, Chapter 1
- Lecture video: [Statistics I Week 1 Playlist](https://www.youtube.com/playlist?list=PLZ2ps__7DhBYrMs3zybOqr1DzMFCX49xG)
- Python libraries: pandas (data types), numpy (numerical operations)

---

**Next Week**: Week 2 - Describing Categorical Data (Frequency distributions, graphs, mode/median for categorical variables)
