
![Build Status](https://www.repostatus.org/badges/latest/concept.svg)

# MFlow artifacts

**All plots that i will put in these readme file you have them available as an MLflow html artifact, so go look at it in the local mlflow UI.**


# Model Results

## Metrics used in the project

During pipeline execution several models were trained and optimized through cross validation, this table shows the results for each of the models.

In the context of pipeline execution and model evaluation, the provided table represents various classification metrics that are commonly used to assess the performance of machine learning models. These metrics help in quantifying how well a classification model is performing on the titanic dataset. Let's expand on each of these metrics and explain their significance:


In the context of pipeline execution and model evaluation, the provided code snippet represents various classification metrics that are commonly used to assess the performance of machine learning models. These metrics help in quantifying how well a classification model is performing on a given dataset. Let's expand on each of these metrics and explain their significance:

1. **Accuracy**:
   - `accuracy_score(y_true, y_pred)` calculates the ratio of correctly predicted instances to the total number of instances in the dataset. It provides an overall measure of the model's correctness in making predictions.

2. **Balanced Accuracy**:
   - `balanced_accuracy_score(y_true, y_pred)` computes the arithmetic mean of sensitivity (true positive rate) and specificity (true negative rate). It is particularly useful when dealing with imbalanced datasets because it considers both positive and negative classes equally.

3. **F1 Score**:
   - `f1_score(y_true, y_pred)` is the harmonic mean of precision and recall. It provides a balance between precision and recall and is particularly useful when you want to find a balance between minimizing false positives and false negatives.

4. **F1 Micro**:
   - `f1_score(y_true, y_pred, average="micro")` computes the F1 score by considering global counts of true positives, false positives, and false negatives. It is suitable for imbalanced datasets.

5. **F1 Macro**:
   - `f1_score(y_true, y_pred, average="macro")` calculates the F1 score separately for each class and then takes the unweighted average (macro-average) of these scores. It gives equal importance to all classes.

6. **F1 Weighted**:
   - `f1_score(y_true, y_pred, average="weighted")` computes the F1 score for each class and then takes a weighted average, where the weights are based on the number of instances in each class. It is useful for imbalanced datasets.

7. **Precision**:
   - `precision_score(y_true, y_pred)` measures the ratio of true positive predictions to the total number of positive predictions. It assesses the model's ability to avoid false positives.

8. **Precision Micro**:
   - `precision_score(y_true, y_pred, average="micro")` calculates precision based on global counts of true positives and false positives.

9. **Precision Macro**:
   - `precision_score(y_true, y_pred, average="macro")` computes precision for each class separately and then takes the unweighted average (macro-average) of these precision scores.

10. **Precision Weighted**:
    - `precision_score(y_true, y_pred, average="weighted")` calculates precision for each class and then takes a weighted average based on class frequencies.

11. **Recall**:
    - `recall_score(y_true, y_pred)` measures the ratio of true positive predictions to the total number of actual positive instances. It assesses the model's ability to find all positive instances.

12. **Recall Micro**:
    - `recall_score(y_true, y_pred, average="micro")` calculates recall based on global counts of true positives and false negatives.

13. **Recall Macro**:
    - `recall_score(y_true, y_pred, average="macro")` computes recall for each class separately and then takes the unweighted average (macro-average) of these recall scores.

14. **Recall Weighted**:
    - `recall_score(y_true, y_pred, average="weighted")` calculates recall for each class and then takes a weighted average based on class frequencies.

15. **Matthews Correlation Coefficient (MCC)**:
    - `matthews_corrcoef(y_true, y_pred)` measures the correlation between predicted and actual binary classifications, considering all four confusion matrix values (true positives, true negatives, false positives, and false negatives). It ranges from -1 (perfect disagreement) to 1 (perfect agreement) with 0 indicating no better than random chance.

16. **Area Under the Curve (AUC)**:
    - AUC represents the area under the Receiver Operating Characteristic (ROC) curve, which evaluates the model's ability to distinguish between positive and negative instances across different probability thresholds. Higher AUC values indicate better model performance, and it is commonly used to assess the overall discriminatory power of a classifier.


The choice of the best metric to evaluate a model on the Titanic dataset depends on the specific goals and priorities of your analysis. So let's put some examples and how I would select the metric.

These are the model results:

![All metrics results](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/results/cross_val_metrics.png)

![Best models visualization](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/results/all_models_graph_results.png)



## Hypotesis and metric selection.

### Survival call.

Let's assume you are a passenger and have to decide whether to buy a ticket for this cruise. In this case, the best metric to choose would be **recall**. Recall measures the model's ability to correctly identify all passengers who will actually survive. A high recall value means that the model has a good probability of detecting most of the passengers who will survive, thus minimizing the risk of buying a ticket and ending up in a dangerous situation.

### Survivor agency

Let's assume you have a travel agency, you know that the barco is not the best, so you need to select what passangers are going to died. So in that case metrics like AUC or MCC would be the best one to select. Where we should prioritize models that are well calibrated with it's probability thresholds.


### No information available

Our case, we don't know apriori the usage of our model, so I would prioritize f1-score, which balance precission and recall of the model. To be more agnostic, I will average all results to select the best model (This is only possible because we have all metrics monotonic increasing, closed to 1 is better)


![Average results](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/results/avg_results_models.png)


In our case our best model is the Xgboost model:

![Average results](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/results/best_pipeline.png)

The pipeline is composed by a k-nn imputer of 20 neighbors, no scaler used (not importart at all, we're using a boosting model), a xgboost model based feature selector of depth 2 and 30 estimators and a xgboost model of depth 6 and 430 estimators. Sounds like a pretty good pipeline, very common on tabular datasets.

# Xgboost model results

## Classification report

The model exhibits superior performance in class 0, which is likely influenced by a slight class imbalance between classes 1 and 0. While this imbalance is not substantial, it could impact the results. Although our best recall score is noteworthy, the overall model performance does not show a significant discrepancy.

![Average results](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/results/classification_report.png)

## Confusion Matrix

Same conclusion regarding the confusion matrix

![confusion_matrix](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/results/confusion_matrix.png)

## ROC-AUC curve

Very good auc, we can change probability thershold to 0.1 in order to maximize our performance, we note this and we should consider this in the model productionalization.

![roc_curve](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/results/roc_curve.png)

## Calibration plot

I've identified a weakness in the calibration plot, indicating that the model lacks proper calibration. To address this issue, it would be beneficial to consider implementing calibration. This can be achieved by refining the `BinaryClassifierSklearnPipeline` class through the inclusion of a CalibratedClassifier wrapper from scikit-learn as the final estimator. This adjustment is particularly crucial in scenarios such as the case of the **Survival Agency**.

![calibration_plot](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/results/calibration_plot.png)


## Cumulative Gain

Calibration should also enhance the cumulative gain plot. Currently, we can capture 80% of survival passengers by targeting 40% of all passengers. However, this targeting ratio should be optimized. Put yourself in the shoes of a consumer enterprise; aiming at 40% of users in a campaign can be quite expensive. Therefore, our goal should be to refine the model until we achieve a lower targeting percentage. For instance, a 25% targeting rate would be more reasonable while still capturing 80% of the positive class.

![cumulative_gain](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/results/cumulative_gain.png)


# Hypertuning

Regarding hypertuning results, the most relevant is that the optimization converged at 300 trials, so 200 trials were mostly unsuccessful, so we can reduce the number of executions trials.

## Converge plot
![hypertuning_curve](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/results/hypertuning_curve.png)


## Hyperparams importance

The most important parameter was: `min_child_weight`  hyperparameter in the XGBoost algorithm, which is used for gradient boosting in decision trees. It represents the minimum sum of instance weight (hessian) needed in a child. It sets a constraint on the minimum amount of data points (samples) required in a leaf (child node) of the tree during the growing process.

The importance of the `min_child_weight` hyperparameter lies in its role in controlling overfitting and influencing the structure of the decision trees within the ensemble. Here's why it's important:

1. **Overfitting Control**: A smaller `min_child_weight` value makes the algorithm more prone to overfitting because it allows the algorithm to create leaf nodes with very few samples. This can lead to excessively deep and complex trees that memorize the training data noise, rather than capturing the underlying patterns. By increasing `min_child_weight`, you constrain the tree from splitting too aggressively, which can help prevent overfitting.

2. **Stability**: Setting an appropriate `min_child_weight` value can lead to more stable and robust models. It reduces the chances of splitting nodes on outliers or noise in the data, making the model's predictions more consistent.

__So in our results meaning that the model is asking us for model exploration in low values, so the optimization is leading into the overfitting problem. As we restrict this value in the exploration space, we're safe, but it's important to pay attention to these conclusions in order to continue reducing or expanding the exploration space, that can lead into a very overffited model.__


![hyperparam_importance](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/results/hyperparam_importance.png)


![rank_plot](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/results/rank_plot.png)


![slice_plot](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/results/slice_plot.png)



# Model interpretability

## Shaps on train dataset
![model_feature_importance](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/results/shap_train.png)

In the model trained, where the most important features are identified as passenger sex, passenger age, passenger ticket, passenger class, and passenger fare, there are some reasons that I can infer why these features might be important:

1. **Passenger Sex**:
   - "Passenger Sex" is a crucial feature because it was one of the first priorities during the evacuation of the Titanic. Women and children were given priority to board lifeboats, which resulted in a higher survival rate for female passengers.

2. **Passenger Age**:
   - "Passenger Age" can be important because, similar to sex, age was a factor in determining priority for lifeboats. Children and the elderly were also given preference, leading to variations in survival rates based on age.

3. **Passenger Ticket**:
   - The "Passenger Ticket" may contain information about cabin or room assignments, which could have had an impact on a passenger's proximity to lifeboats or their access to escape routes. Some tickets may have indicated higher-deck cabins, which might have been closer to lifeboat stations. Also checking direction on the model this could affect that tickets that has a lower value could be prioritized, also could be related to the passenger class inside the boat.

4. **Passenger Class**:
   - "Passenger Class" is significant because it is closely related to socio-economic status. Passengers in higher classes (e.g., first class) may have had better access to lifeboats and superior accommodations compared to passengers in lower classes. This class-based disparity is well-documented in the Titanic disaster.

5. **Passenger Fare**:
   - "Passenger Fare" is correlated with passenger class and socio-economic status. Higher-fare passengers were more likely to be in first class and therefore had better access to safety measures.

6. **Number of Family Members Onboard (Family Size)**:
   - The total number of family members on the Titanic can be a valuable feature. It could help capture the dynamics of passengers traveling with family members, which might influence their survival. For example, families might have stayed together or sought safety together, potentially impacting survival rates.

7. **Unknown Cabin Values**:
   - Accounting for unknown cabin values is crucial because a significant portion of the cabin information in the Titanic dataset is missing (often denoted as "NaN" or "Unknown"). The absence of cabin information could itself be a useful indicator or might be related to certain passenger characteristics. By including this feature, I acknowledge and account for the lack of information, which can help your model make predictions for passengers with missing cabin data more effectively. This feature said something about the people that knows the cabin were they stayed in the boat


## Model feature importance train
![model_feature_importance](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/results/model_feature_importance.png)

## Shap feature importance train
![shap_feature_importance](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/results/shap_feature_importance.png)


We denote a difference between Gini importance and Shapley (SHAP) importance. This disparity arises primarily because the Gini importance and SHAP importance are based on different principles and approaches when assessing feature importance in machine learning.

**Gini Importance**:
- Gini importance is a metric derived from decision tree-based algorithms, such as Random Forests or Gradient Boosting. It measures the feature's contribution to the reduction in impurity (often Gini impurity) within the tree nodes. Essentially, Gini importance quantifies how much a feature helps in splitting the data into pure or homogenous classes. Higher Gini importance suggests that a feature is more effective in making splits that separate classes.

**Shapley Importance (SHAP)**:
- SHAP importance, on the other hand, it provides a more comprehensive and theoretically sound way to attribute the contribution of each feature to a model's prediction for a specific instance. SHAP values consider all possible combinations of features and measure how much each feature adds or subtracts from the model's prediction when combined with other features. This approach accounts for feature interactions and provides a more nuanced understanding of feature importance, especially in complex models like gradient boosting or deep learning.

__We note this as a lack of interpretability, we should try to get simular values, so the model becomes more transparent__


## Shaps on test dataset

I compute shap values to check if there is an important dataset drift on the test dataset, these is not the case, shaps are equivalent on both dataset. So nice to check.

![model_feature_importance](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/results/shap_test.png)



## Model predictive control explanations


Let's go deeper on interpretability and its relevance in the context of maximizing survival probability on the Titanic dataset requires a deeper understanding of the key features and strategies involved. In this scenario, Model Predictive Control (MPC) theory can offer valuable insights into defining effective survival strategies.

1. **Interpretability and Survival Probability**:
   - Interpretability in machine learning refers to the ability to understand and explain the decisions made by a model. In the context of maximizing survival probability on the Titanic, interpretability becomes crucial because it allows us to identify the most important features contributing to survival outcomes and define effective strategies for passenger survival.

2. **Defining Survival Strategies**:
   - MPC theory plays a significant role in defining survival strategies. MPC is a control strategy that optimizes decisions in real-time based on a model of the system and a specific objective. In the context of Titanic survival, an MPC-based strategy involves continuously updating decisions to maximize the probability of survival as the situation unfolds.


By leveraging MPC theory, we can create a dynamic and adaptable survival strategy that takes into account the importance of various features, updates decisions in real-time, and maximizes the chances of passenger survival. Interpretability techniques help us understand the underlying model, the significance of each feature, and how these features influence the strategy. This holistic approach can be instrumental in decision-making processes, especially in situations where lives are at stake.


![mpc_importance](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/results/mpc_importance.png)


![slice_search_space.ng](https://github.com/matheus695p/titanic-dataset/blob/main/docs/images/results/slice_search_space.ng.png)


So for example in this case, let's look for a strategy that will save the most passengers.

Creating a survival strategy where "Sex," "Class," and "Fare" are the most important factors involves prioritizing passengers based on these features to maximize their chances of survival. Here's a survival strategy focusing on these key factors:

**Survival Strategy: Prioritizing Based on Sex, Class, and Fare**

1. **Priority for Women and Children**:
   - Priority is given to women and children, as historically, they were among the first to board lifeboats during the Titanic disaster. This principle prioritizes female passengers (regardless of class) and children in all classes, so in the case of we're an agency, what i would do, would be to create promo tickets for women and children. (Imagine this is a churn problem, this sounds like a good commercial strategy)

2. **First-Class Passengers**:
   - Among adult male passengers, priority is given to first-class passengers. First-class passengers are directed to lifeboats before other male passengers in lower classes. This reflects the historical practice of prioritizing higher-class passengers. So in this case I would advice male passengers to buy the most expensive ticket (jaja just hypothesis no judgement haah)

3. **Fare Adjustment for Families**:
   - To ensure families stay together, if an adult with children purchased a lower-class ticket, they are allowed to board the lifeboat with their family members, regardless of the class-based priority. This helps maintain family units during evacuation.

4. **Continuous Monitoring and Adaptation**:
   - Throughout the evacuation process, the strategy is continuously monitored and adapted based on the evolving situation. This may involve reallocating resources, reassessing priorities, and ensuring that the most vulnerable individuals (women and children) are protected.

5. **Resource Allocation**:
   - Allocate resources such as lifeboats, life vests, and crew assistance based on the priority order defined by sex, class, and fare.

6.  **Continuous Improvement**:
    - After the evacuation, conduct a thorough review of the strategy's effectiveness and identify areas for improvement. Lessons learned can be used to enhance future survival strategies.

This survival strategy aims to maximize the chances of survival by prioritizing passengers based on historical patterns and key factors such as sex, class, and fare. It is essential to note that the strategy is designed for educational and historical context.

This are just hypotheses, but is the same reasoning on any machine learning problem, we should always focus on the impact and strategies that our models can provide.


Happy hacking !

Matheus,
