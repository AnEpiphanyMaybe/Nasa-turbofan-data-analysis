# NASA CMAPSS Remaining Useful Life Prediction

## Overview

This project investigates the prediction of Remaining Useful Life (RUL)
for turbofan engines using the NASA CMAPSS dataset. The goal was to
compare several machine learning and deep learning approaches and
investigate which sensor measurements and temporal patterns provide the
most useful information about engine degradation.

Models explored include:

- Linear Regression with Ridge, Lasso, and Elastic Net regularization
- Random Forest
- XGBoost Regressor
- XGBoost Random Forest
- Long Short-Term Memory (LSTM) network
- 1D Convolutional Neural Network (CNN)

The project also investigates feature engineering approaches including
rolling statistics, sequence windows, slope features, and RUL capping.

## Dataset

The NASA CMAPSS dataset contains simulated turbofan engine run-to-failure
data. Each engine contains measurements from multiple sensors over a
sequence of operating cycles.

Remaining Useful Life was calculated from the number of cycles remaining
before engine failure.

Exploratory analysis revealed substantial multicollinearity among several
sensor measurements, as well as nonlinear relationships between sensor
measurements and RUL.

## Methodology

### Linear Models

Multiple linear regression was initially explored as an interpretable
baseline. Strong multicollinearity between sensors made ordinary linear
regression problematic, so Ridge, Lasso, and Elastic Net regularization
were evaluated.

Ridge and Lasso produced similar performance, with R² values of
approximately 0.47. Elastic Net achieved approximately 0.46.

These results suggested that regularization could address some
multicollinearity but that the relationship between the available
features and RUL was not adequately captured by a linear model.

### Random Forest

Random Forest was introduced to capture nonlinear relationships in the
sensor data. Hyperparameters were selected using GridSearchCV with
group-aware cross-validation to prevent measurements from the same
engine from being improperly distributed across folds.

The baseline Random Forest achieved approximately:

- R²: 0.52
- MSE: 1409

Sensors 11, 9, 12, 4, and 14 were among the most informative features.

Rolling means and standard deviations were also tested but did not
improve predictive performance, suggesting that the tree model could
already recover much of the useful trend information from the raw
measurements.

### XGBoost

XGBoost was tested to determine whether sequential boosting could improve
on the Random Forest model.

The XGBoost Regressor achieved approximately:

- R²: 0.529
- RMSE: 40.49

A held-out GroupShuffleSplit validation set with early stopping produced
similar results (R² ≈ 0.527, RMSE ≈ 40.57), providing evidence that the
model's performance generalized beyond the training data.

Rolling means and standard deviations using windows of 5, 10, and 20
cycles did not meaningfully improve performance.

Slope features were also investigated but produced unstable extreme
values and substantial overfitting. A percent-of-life-elapsed feature
was removed because its definition introduced a structural mismatch
between the run-to-failure training data and truncated test engines.

### LSTM

An LSTM was evaluated to determine whether longer temporal dependencies
could improve RUL prediction.

Unexpectedly, the model performed best with a window size of 1.
Increasing sequence length did not improve performance, suggesting that
most of the predictive information available to this model was contained
in the current sensor state rather than longer sequence history.

Final performance was approximately:

- R²: 0.521
- RMSE: 40.82

Permutation importance identified sensors 7, 20, 4, 2, and 15 as the
most influential features.

### 1D CNN

A 1D CNN was tested to identify local temporal patterns in sensor
measurements. The final architecture used two convolutional layers with
ReLU activation and the Adam optimizer.

A 30-cycle window produced approximately:

- R²: 0.518
- RMSE: 37.90

Larger windows appeared to improve R², but further investigation showed
that increasing the window size excluded shorter-lived engines from
evaluation. This changed the evaluation population and created an
artificial improvement rather than demonstrating genuine model progress.

Permutation importance identified sensors 20, 14, 11, 21, and 7 as
particularly influential.

## Key Findings

### 1. Model performance reached a similar ceiling

Despite substantial differences in model complexity, most nonlinear
models produced similar predictive performance. More complex temporal
models did not produce a clear improvement over tree-based approaches.

### 2. Longer temporal history provided limited additional information

Rolling means, rolling standard deviations, slope features, and longer
LSTM windows generally failed to improve performance. This suggests that
much of the useful degradation information is already represented by the
current sensor state.

### 3. Several sensors consistently contained useful degradation signals

Sensors 9, 14, 7, and 11 repeatedly appeared among important features,
although importance varied by model.

Correlation analysis revealed two particularly strong relationships:

- Sensors 9 and 14: r ≈ 0.96
- Sensors 7 and 11: r ≈ -0.82

The two pairs were only weakly correlated with one another, suggesting
that each pair contains substantial internal redundancy while the two
pairs capture different aspects of engine degradation.

### 4. Feature importance depends on the model and importance method

Tree models and neural networks did not always identify the same sensors
as important. Tree-based importance and permutation importance measure
different aspects of model behavior, so their magnitudes should not be
interpreted as directly comparable.

## Model Comparison

| Model | R² | Error Metric |
|---|---:|---:|
| Ridge Regression | ~0.47 | MSE ~1840.72 |
| Lasso Regression | ~0.47 | MSE ~1839.79 |
| Elastic Net | ~0.46 | MSE ~1862.91 |
| Random Forest | ~0.52 | MSE ~1409.18 |
| XGBoost Regressor | ~0.529 | RMSE ~40.49 |
| XGBoost Random Forest | ~0.524 | RMSE ~40.67 |
| LSTM | ~0.521 | RMSE ~40.82 |
| 1D CNN | ~0.518 | RMSE ~37.90 |

> Note: Some experiments used MSE while later experiments report RMSE,
> so error values should only be compared when the same metric and
> evaluation setup were used.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Matplotlib

## Future Work

Future work could explore ensemble architectures combining models that
appear to capture different aspects of the degradation signal. For
example, CNN or LSTM representations could potentially be combined with
tree-based predictions.

Additional work could also investigate the physical meaning of the
anonymized sensor measurements and determine whether domain-specific
feature engineering can extract additional degradation information.








Data Set: FD001
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: ONE (HPC Degradation)

Data Set: FD002
Train trjectories: 260
Test trajectories: 259
Conditions: SIX 
Fault Modes: ONE (HPC Degradation)

Data Set: FD003
Train trjectories: 100
Test trajectories: 100
Conditions: ONE (Sea Level)
Fault Modes: TWO (HPC Degradation, Fan Degradation)

Data Set: FD004
Train trjectories: 248
Test trajectories: 249
Conditions: SIX 
Fault Modes: TWO (HPC Degradation, Fan Degradation)



Experimental Scenario

Data sets consists of multiple multivariate time series. Each data set is further divided into training and test subsets. Each time series is from a different engine – i.e., the data can be considered to be from a fleet of engines of the same type. Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user. This wear and variation is considered normal, i.e., it is not considered a fault condition. There are three operational settings that have a substantial effect on engine performance. These settings are also included in the data. The data is contaminated with sensor noise.

The engine is operating normally at the start of each time series, and develops a fault at some point during the series. In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends some time prior to system failure. The objective of the competition is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of operational cycles after the last cycle that the engine will continue to operate. Also provided a vector of true Remaining Useful Life (RUL) values for the test data.

The data are provided as a zip-compressed text file with 26 columns of numbers, separated by spaces. Each row is a snapshot of data taken during a single operational cycle, each column is a different variable. The columns correspond to:
1)	unit number
2)	time, in cycles
3)	operational setting 1
4)	operational setting 2
5)	operational setting 3
6)	sensor measurement  1
7)	sensor measurement  2
...
26)	sensor measurement  26


Reference: A. Saxena, K. Goebel, D. Simon, and N. Eklund, “Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation”, in the Proceedings of the Ist International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
