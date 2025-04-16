# Obesity Level Classification Project

This project aims to build a machine learning model capable of classifying an individual's obesity level based on demographic indicators and lifestyle habits (e.g., age, gender, BMI, physical activity level, dietary patterns, etc.).

The data was thoroughly preprocessed and analyzed using modern data visualization tools. Subsequently, multiple machine learning models were trained and compared to identify the algorithm with the highest accuracy for this classification problem.

  # Key Features:
- Data Preprocessing: Cleaning, label encoding, and feature scaling
- Exploratory Data Analysis (EDA): Identifying trends, distributions, and relationships between variables
- Model Training: Implemented Decision Tree, Random Forest, KNN, Logistic Regression, SVM, and XGBoost
- Model Evaluation: Compared accuracy scores and confusion matrices
- Conclusion: Selected the best-performing model and identified the most important features

# Processing
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Age</th>
      <th>Height</th>
      <th>Weight</th>
      <th>FCVC</th>
      <th>NCP</th>
      <th>CH2O</th>
      <th>FAF</th>
      <th>TUE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20758.00000</td>
      <td>20758.000000</td>
      <td>20758.000000</td>
      <td>20758.000000</td>
      <td>20758.000000</td>
      <td>20758.000000</td>
      <td>20758.000000</td>
      <td>20758.000000</td>
      <td>20758.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>10378.50000</td>
      <td>23.841804</td>
      <td>1.700245</td>
      <td>87.887768</td>
      <td>2.445908</td>
      <td>2.761332</td>
      <td>2.029418</td>
      <td>0.981747</td>
      <td>0.616756</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5992.46278</td>
      <td>5.688072</td>
      <td>0.087312</td>
      <td>26.379443</td>
      <td>0.533218</td>
      <td>0.705375</td>
      <td>0.608467</td>
      <td>0.838302</td>
      <td>0.602113</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00000</td>
      <td>14.000000</td>
      <td>1.450000</td>
      <td>39.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5189.25000</td>
      <td>20.000000</td>
      <td>1.631856</td>
      <td>66.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>1.792022</td>
      <td>0.008013</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>10378.50000</td>
      <td>22.815416</td>
      <td>1.700000</td>
      <td>84.064875</td>
      <td>2.393837</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.573887</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>15567.75000</td>
      <td>26.000000</td>
      <td>1.762887</td>
      <td>111.600553</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.549617</td>
      <td>1.587406</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>20757.00000</td>
      <td>61.000000</td>
      <td>1.975663</td>
      <td>165.057269</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
    </tr>
  </tbody>
</table>
</div>

- Number of samples: 20,758. All columns are fully populated (no missing values)
- Age:
  - Mean: 23.84 years old,
  - Range: 14 to 61.
  - Age distribution is skewed toward younger individuals (25% under 20, 75% under 26) → the dataset is concentrated in the teen and young adult age groups.
- Height:
  - Average: 1.70 m – reasonable values
  - Range: 1.45 m to 1.98 m
  - Distribution is fairly normal with a low standard deviation (±0.087 m)
- Weight:
  - Average: 87.88 kg – relatively high
  - Min - Max: 39 kg – 165 kg → wide range, high standard deviation (±26.38 kg)
  - Likely right-skewed distribution (many overweight or obese individuals)
-  **Lifestyle & Eating Habits**
- FCVC (Frequency of vegetable consumption):
  - Average: 2.45 (on a scale from 1 to 3) → most people consume vegetables regularly
  - Majority score between 2 and 3
- NCP (Number of main meals per day):
  - Average: 2.76, median = 3 → most people eat 3 meals a day    
  - Evenly distributed (min = 1, max = 4)
- CH2O (Daily water intake):
  - Average: ~2.03 (scale 1–3) → most people drink an adequate amount of water daily
- FAF (Weekly physical activity frequency):
  - Average: ~0.98, but high standard deviation → many people are inactive
  - Range: 0 to 3 → some individuals are completely inactive
- TUE (Time spent using technology for work or study):
  - Average: ~0.62 (on a 0–2 scale)
  - Median: ~0.57 → most people spend a relatively small amount of time with tech
  - Values range from 0 to 2

 # EDA
![image](https://github.com/user-attachments/assets/3a1bed0a-434d-488a-b490-4231990a4a40)
- The class distribution is relatively balanced, with no extreme class imbalance, which is beneficial for training classification models.
![image](https://github.com/user-attachments/assets/cd639612-3cb0-412e-b91e-76003c8fe494)
- **Demographic Information**
  - Gender: The distribution is nearly equal between males and females → no significant gender imbalance.
  - Age: Right-skewed distribution, with most individuals aged 14–30 → the data leans toward a younger population.
  - Height & Weight: Show multimodal and diverse distributions, reflecting a wide variety of body types.

- **Family History & Eating Habits**
  - family_history_with_overweight: Most participants have a family history of overweight → genetic factors are prevalent.
  - FAVC (frequent consumption of high-calorie food): Majority answered "Yes" → unhealthy eating habits are common.
  - FCVC (vegetable consumption level): Mostly in the 2–3 range → participants maintain a relatively healthy intake of vegetables.
  - NCP (number of main meals/day): Most eat 3 meals a day, aligning with nutritional recommendations.

- **Snacking & Smoking Behavior**
    - CAEC (eating between meals): Mostly "Sometimes", with some "Frequently" → snacking is present but not dominant.
    - SMOKE: Very few people smoke → the sample largely consists of non-smokers.

- **Hydration & Calorie Awareness**
  - CH2O (daily water intake): Concentrated around 2 liters/day, which is fairly healthy.
  - SCC (calorie monitoring): Mostly "No" → most people do not track their calorie intake.

- **Physical Activity & Tech Usage**
  - FAF (physical activity per week): Most people are in the 0–1 times/week range, indicating low activity levels → potential risk for overweight.
  - TUE (technology usage time): Peaks at both low (0–0.5) and high (2.0) → polarized behavior in tech usage.

- **Drinking & Transportation Habits**
  - CALC (alcohol consumption): Mostly "Sometimes", followed by "No" → infrequent drinking is common.
  - MTRANS (main transportation mode): Public transport is dominant → could be influenced by urban settings or student lifestyle.

- **Summary**:
  
  The dataset clearly reflects the behavioral and physical characteristics of modern youth: low physical activity, unbalanced eating habits, high tech usage, but limited smoking and moderate alcohol use.
  It is a well-suited dataset for building predictive models of obesity based on lifestyle and demographic features.


# Aplly model
- Logistic Regression  
  ![image](https://github.com/user-attachments/assets/83051a24-7ad1-4e47-875e-74813beb0962)

- Decision Tree  
  ![image](https://github.com/user-attachments/assets/e7c4b98d-bb3d-4305-9824-616006368911)

- KNN  
  ![image](https://github.com/user-attachments/assets/25918183-9a4e-4819-b61e-5c4e3bf9c787)

- SVM  
  ![image](https://github.com/user-attachments/assets/cd936ecc-cc7a-4a5a-a0ca-e95b123d4756)

- RandomForest  
  ![image](https://github.com/user-attachments/assets/50376d63-a750-45f1-ae86-a650dc4209e7)

- XGBoost  
  ![image](https://github.com/user-attachments/assets/5b761da6-0787-42cd-aa7e-5abe90326964)

- Adaboost  
  ![image](https://github.com/user-attachments/assets/253a7827-6506-47b0-923d-24570991d2dd)

# Model Evaluation 
![image](https://github.com/user-attachments/assets/7e4017e2-c87c-4f5f-968e-57a920386ee1)
- Choose the 3 algorithms with the highest accuracy to perform hyperparameter tuning.

# Hyper-parameter Tuning
To improve model performance, we applied Grid Search to find the optimal hyperparameters for the top three models (based on accuracy): XGBoost, Random Forest, and KNN.

- XGBoost
  - Before tuning: Accuracy = 0.9101
  - After tuning: Accuracy = 0.9135
  - Best Parameters:colsample_bytree: 0.6, learning_rate: 0.2,max_depth: 3, n_estimators: 200,
  subsample: 0.8

- Random Forest
  - Before tuning: Accuracy = 0.9061
  - After tuning: Accuracy = 0.9090
  - Best Parameters:n_estimators: 200, max_features: 'log2', max_depth: None, min_samples_split: 10, min_samples_leaf: 2, bootstrap: False
- K-Nearest Neighbors (KNN)
  - Before tuning: Accuracy = 0.8510
  - After tuning: Accuracy = 0.8709
  - Best Parameters:n_neighbors: 7, weights: 'uniform', p: 2 (Euclidean distance)
# Conclusion:
Within the scope of this study, XGBoost emerged as the most outstanding model, demonstrating superior performance compared to the other models. The optimization of its hyperparameters significantly enhanced the model's effectiveness. This not only improved prediction quality but also strengthened its potential for real-world applications. Furthermore, the study revealed the complex relationships between various factors related to physical condition and daily lifestyle, such as gender, age, family history, dietary habits, physical activity level, and more. Therefore, a multidimensional intervention approach is required—focusing on comprehensive lifestyle improvements rather than targeting any single factor—to effectively prevent and manage obesity in individuals.
