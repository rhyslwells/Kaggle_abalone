To do:

# Preprocessing

# EDA 

Target Analysis 

Numericals - else

Corrolation: 
which features are corrolated with the target - if all - keep all.
which features are corrolated with each other: 
    - if two features are highly corrolated, we can drop one of them
    - if two features are highly corrolated with the target, we can keep both

For each numerical:
Histograms
Boxplots
Statistical data : what to look for:

- mean and median difference implies there are outliers
- skewness and kurtosis measure the shape of the distribution by looking at the tails of the distribution and the peak of the distribution.

Handle outliers 

Log transformation to help the skewness.

Categoricals - Sex

sns.pairplot

for each numerical boxplot for sex.

filter outliers again per sex and numerical?

Idenifity features which have distinction in heights between boxplots. - apparently none.

encode to numericals - one hot encoding

Ask gpt for more things to consider.

Do statistical tests to see if the difference in means is significant.

Height as histogram - two overlapping histograms (M & F). 

Is height a good predictor of sex are these distrubutions statitically different? chi square, kstest, ttest.


# Feature engineering:

