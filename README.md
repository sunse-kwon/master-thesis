# Master-thesis

The code is for a thesis of the MSc of Data Science and Society at Tilburg University, Netherlands titled "Multistep short-term parking forecasting with ensemble and deep learning models".

## Data
- Two datasets in the experiment are real-time residential parking lot data and parking facility data to the corresponding parking lot.

### Time-series data
- Real-time data could be retrieved from API every minute. In this study, data was collected from 22-09-2022 to 01-11-2022 in 15-minute intervals and stored in CSV format.
- The total size of the raw data was 10072795 rows and six columns. A total 1968 parking lots were collected from the sensors installed in each location.
<img width="742" alt="image" src="https://github.com/sunse-kwon/master-thesis/assets/94329884/effb021e-43da-4a3a-a3d5-2b921dc98c2b">

### Tabular data
- It shows each parking lot’s facility information which is available in CSV format.
<img width="744" alt="image" src="https://github.com/sunse-kwon/master-thesis/assets/94329884/29e96c30-d6b7-459d-84ff-2a24743f4127">

## Algorithms
- Bagging Regressor
- XGBoost Regressor
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Sequence to Sequence (Seq2seq) - > used LSTM and GRU cells

## Results
- The result below is to answer research questions defined in the thesis. check the original thesis paper for further details.

### Effect of features
- Two datasets were used to test the effect of features. univariate and multivariate datasets.
- For selecting features for the multivariate dataset, RFE(recursive feature elimination) from sklearn package was used to select half of the entire feature set(24 features).
- Then, four models (Bagging regressor, XGBoost regressor, LSTM, GRU), which were trained with time window(used single previous timestep to predict next value) used for comparison.
- As can be seen, multivariate models have demonstrated better performance.
<img width="696" alt="image" src="https://github.com/sunse-kwon/master-thesis/assets/94329884/0ebd5729-ddc4-4ea8-8798-ced15787d93c">


### Performance on Different Time Windows
- Models were tested on window 1, window 2, and window 3 (see section 3.2.7 in the thesis paper) to determine the appropriate time window size. Those three candidates were selected based on EDA autocorrelation analysis.
- The expected suitable window size would be window 1 for 855 parking lots, while window 3 would be applicable for an individual parking lot.
- To answer RQ4, models were tested in five different regions with different subset levels(group level: 10 to 12 parking lots per region, individual level: 1 parking lot per region). Those regions were split based on 5-fold k-mean clustering using x,y coordinates.
- When using the whole sample, the model's performance did not vary across window sizes. Nevertheless, when the sample size decreases smaller scale, the performance of models varies based on different window sizes and regions.
- Based on group and individual level analysis in five regions, East and West areas showed a drastic change of performance across time windows, while central, south, and north areas were not much affected by time windows. It implies that the west and east areas have more non-stationary patterns than other regions, making prediction more problematic.
- In that sense, the study determined the appropriate window size would be window 3 (use 16 previous timesteps to predict the next timestep ahead). The reason is the robustness of the model’s performance at a different sample size. When window 1, the performance of models varied when decreasing the size of samples. In contrast, the performance of models was stable across the different sizes of samples in window 3.
  
#### Entire dataset
<img width="690" alt="image" src="https://github.com/sunse-kwon/master-thesis/assets/94329884/9b1ad1f1-76c3-4a13-b961-17f627488222">

#### Group level 
<img width="688" alt="image" src="https://github.com/sunse-kwon/master-thesis/assets/94329884/4654a1a3-3569-49ff-8500-8d45705e0ea0">

#### Individual level
<img width="691" alt="image" src="https://github.com/sunse-kwon/master-thesis/assets/94329884/9be9bbae-a7a6-43a3-b432-2dde9604b4ed">

### Performance on Different Time Horizon

- Six models were tested on the first 40 consecutive time horizons on the test-set. As mentioned previous section, To answer RQ4, models were tested in five different regions with different subset levels(group level: 10 to 12 parking lots per region, individual level: 1 parking lot per region).
- To generate a 40-time horizon, the study used a recursive strategy for four models(Bagging, XGBoost, LSTM, GRU) which used a sliding window technique using the output of models and iterated 40 times. and added Seq2seq models to conduct recursive strategy with multioutput. (we called it a hybrid recursive-multioutput strategy in convenience).
- There was a regional difference between east and west against other areas, as east and west exhibited higher error scores. The hypothesis of higher prediction error in the east and west areas would be that those regions have characteristics such as longer travel time to the central area and less complexity of public transport network in comparison to other areas. 
- There was a superiority of models per size of sample and different regions. For the whole sample level, the bagging regressor exhibited stable performance across the whole time horizon, but in general, the hybrid strategy was superior to the recursive strategy. Even though the bagging regressor used a recursive strategy, bagging produced robust prediction results in multistep forecasting compared to other recursive strategy models, as they displayed a drastic drop in performance after 5-step horizon.
- One possible reason for the superiority of seq2seq models could be that multioutput sequence generation preserves temporal dependency well in the short term. Another possible reason could be that the number of iterations to produce the entire horizon was much less(10 iterations) than the pure recursive strategy(40 iterations).
- When comparing group and individual level analysis, while bagging regressor demonstrated the highest performance in the east and south areas, seq2seq models had superiority over other areas. (central, west, north).

#### Entire dataset
<img width="694" alt="image" src="https://github.com/sunse-kwon/master-thesis/assets/94329884/881e783c-5e02-4600-ab24-acb488de6d71">

#### Group level 
![image](https://github.com/sunse-kwon/master-thesis/assets/94329884/6367bb03-0081-4d18-a6b7-71829058413e)

#### Individual level
![image](https://github.com/sunse-kwon/master-thesis/assets/94329884/784ba988-aae0-44cc-a5ac-d51fdb2a3200)
