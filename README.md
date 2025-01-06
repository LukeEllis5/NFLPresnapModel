## Run/Pass Pre-snap Prediction
### Metric Track

### Overview
As advancements in data collection in the NFL have continued to grow, so has the adoption of data science to make predictions. The following model was built to give NFL defenses an advantage against the San Francisco 49ers by producing a probability that the next offensive play will be a designed run or pass play based on their tendencies based on pre-play line-up and game situation. By using this model, defensive coaches will be able to call an audible that will give them an advantage.

### Data Source
The data was supplied by the NFL and was from weeks 1-6 of the 2023 season. The training data was compressed using gameID and playID and was focused on San Francisco's offensive plays. QB kneels were removed from the dataset. The final dataset contained 451 plays and seven features.

### Target Variable
The target variable is a Boolean feature that has a value of (1) if it is a designed running play or (0) if it was a pass play. To put an emphasis on play calling, QB scrambles do not count as designed runs.

### Model
The model is a random forest classifier that uses seven features to produce the probability that the play will be a designed run play. Because of the imbalance of run to pass plays (194-257), the model uses class weights to prevent biases since there is an imbalance in the data.

The model performed well with an accuracy of 80.9% and a ROC AUC of 87.4%. True positives were sought out due to the costly nature of calling the wrong play when anticipating a run play. Because of this, a high ROC AUC and precision were sought out with the model identifying 81% for pass plays and 79% for run plays.

### Feature Selection
The following features were selected based on factors that could influence an offense's play-calling. The definitions are from the NFL's Big Data Bowl Dataset Description. The SHAP values represent the relative importance of each feature in making a model's prediction, with higher values indicating greater impact; however each feature’s SHAP value reflects its individual contribution to the model’s output rather than an overall total.

offenseFormation (SHAP 17.76%): Formation used by the possession team. Offensive formations are categorical data that have six options: SHOTGUN, SINGLEBACK, I_FORM, EMPTY, PISTOL, and JUMBO. Offensive formations originally had thirteen null values; these nulls were dropped. The data went through OneHotEncoding.

motionSinceLineset (SHAP: 12.24%): Boolean indicating whether the player went in motion after they were initially set at the line on this play. The initial dataset had 1,480 null values; because the value is important the nulls were filled using a KNN imputer. This feature went through a Binary Encoder before being passed into the training data.

receiverAlignment (SHAP 8.81%): Enumerated as 0x0, 1x0, 1x1, 2x0, 2x1, 2x2, 3x0, 3x1, 3x2. It is categorical data that underwent OneHotEncoding.

presnap_score_difference (SHAP 3.79%): This feature was created to capture potential play-calling tendencies based on the score difference. If San Francisco is up, the number will be positive; if they are down, the number will be negative. This went through StandardScaler.

down_yardsToGo (SHAP 3.54%): This feature was created by taking the down and multiplying it with the yards to go to create an expanded scale. This allows for a wider range, punishing later downs with further yards to go. This feature went through StandardScaler.

gameClock_seconds (SHAP 2.04%): From the data feature gameClock, it is the time on the clock of the play (MM:SS). This was converted into seconds and then went through StandardScaler.

shiftSinceLineset (SHAP 2.00%): Boolean indicating whether the player shifted since the lineset; Rule: Each player has their own lineset moment, and whether they shift is based on if they move more than 2.5 yards from where they were at their lineset moment. There were thirteen null values; because of the small amount, the null values were dropped. This feature went through a Binary Encoder before being passed into the training data.


We performed a simple correlation to the play result being a run play. As you can see, features that would indicate a run had a higher correlation than features that would indicate a pass.

##### Correlation with playResult:
playResult                     1.000000                                      
offenseFormation_I_FORM        0.407478                                   
receiverAlignment_2x1          0.384195                                   
presnap_score_difference       0.282033                                   
offenseFormation_SINGLEBACK    0.227010                                   
shiftSinceLineset              0.217943                                   
offenseFormation_PISTOL        0.144669                                   
offenseFormation_JUMBO         0.108878                                   
receiverAlignment_1x1          0.076817                                   
gameClock_seconds              0.063792                                   
receiverAlignment_2x2          0.012063                                   
receiverAlignment_4x1         -0.057986                                   
receiverAlignment_3x1         -0.172361                                   
receiverAlignment_3x2         -0.270576                                   
offenseFormation_EMPTY        -0.278012                                   
motionSinceLineset            -0.286902                                   
down_yardsToGo                -0.348579                                   
offenseFormation_SHOTGUN      -0.379380                                   

### Hyperparameters
The model underwent an iterative process to fine-tune the hyperparameter ranges, optimizing for the best ROC AUC and precision scores. Using grid search, the model was tuned to achieve peak performance. The best hyperparameters were below:

classifier__bootstrap : False
classifier__max_depth: 10                                          
classifier__max_features: 'sqrt'                               
classifier__min_samples_leaf: 2                             
classifier__min_samples_split: 25                   
classifier__n_estimators: 300              

To mitigate potential overfitting, cross-validation scores were closely monitored. The mean cross-validation score was 88.4% indicating a strong performance. The low standard deviation of 2.6% shows a low variability in the model's performance across folds. 


### Model Performance
The model overall performed well with an accuracy score of 80.9% and a ROC AUC of 87.4%. The confusion matrix below will show that the model was able to accurately identify between run and pass plays. Recall was higher at 84% for pass plays compared to 76% for run plays. The f1-score was 83% for pass plays and 78% for run plays. These results allow the defense to have enough confidence in the model where it could give them a competitive advantage.

Classification Report:

                precision    recall  f1-score   support

         Pass       0.82      0.84      0.83        77
         Run        0.79      0.76      0.78        59



![image.png](attachment:a2abe651-cbcf-4ed7-9651-6f92d84f900b.png)

![image.png](attachment:9f2728a9-0b03-41d9-8456-3a71df6e2a77.png)

The learning curve graph below shows the relationship between the model performance and the size of the training data. As the training data increases, the model fits the training data well. Because there is a gap between the training data and cross validation score, this may suggest in some bias in the model. This may indicate slight overfitting in the model. The overfitting could benefit by additional training data. Both lines do tend to level out which would indicate additional training data would have a slight benefit.

![image.png](attachment:570a531a-1bcf-4ed2-9177-88dec836fcdf.png)


### Limitations and Assumptions
Although the model's metrics are strong, there is room for improvement. To maintain accuracy, the model should be retrained every week to capture any new tendencies the San Francisco 49ers might adopt. As the season goes on, the model will need to start dropping earlier weeks of data from the training set or have a greater emphasis on more recent weeks. The model could improve by including features that account for player personnel, such as whether key players like Christian McCaffrey are in the game, as this could influence play-calling tendencies.
