# Explainable-speech-ML-models

CSCI 5622 - Assignment 4

Members : 
Aidan Bagley,
Dheeraj Gajula,
John Maddox,
Rahul Ganesan


Aidan and Rahul Meeting 11/20:

## Todo:
* Merge Prosodic and Textual datasets into cohesive data cleaning and feature selection --> select features just corresponding overall, just corresponding to excited, corresponding to joint overall and excited --> three feature sets per dataset
*   Consistency in Particpant to Interview mapping
  
* Trained models for textual dataset for just overall, just excited, both overall, and excited simeltanously --> three feature sets --> three models (one set per model)
* Trained models for prosodic dataset for just overall, just excited, both overall, and excited simeltanously --> three feature sets --> three models (one set per model)
* Joint models for just overall, just excited, both overall, and excited simeltanously --> three joint feature sets --> three joint mega-models or 6 independent models (1-to-1) with averaged outputs
*   Do we have average of output for each model or one mega-model? Could also weighted average based on confidence, correlation, etc

* John create overleaf and presentation skeleton

* Explainablitity: meet tomorrow to discuss
* Pre-trained LLM: meet tomorrow to discuss


## Thoughts:
* Random seed = 42 for train test split
* Feature selection using just training data - not testing data
* GridSearchCV for model hyperparameters selection
* How to best combine multiple models --> Do we bootstrap our models and do majority/avergae voting? Do we train one mega model on all features from each model? Train each option for either just excited, just overall, or joint excited + overall
* Not too much hyperparameter tuning if model is good enough

---

* Feature Sets for both datasets:
  * K-Best for just excitement
  * K-Best for just overall
  * K-Best for joint [overall, excitement]

* Independent Models:
*   Textual:
      * Just excitement
      * Just overall
      * Joint [overall, excitement]
*   Prosodic for average dataset:
      * Just excitement
      * Just overall
      * Joint [overall, excitement]
*   Prosodic for all questions dataset:
      * Just excitement
      * Just overall
      * Joint [overall, excitement]

* Multi-Modal Models
  * Aggregate Feature Set Models
    * Textual Features + Prosodic Features trained to output single overall score 
    * Textual Features + Prosodic Features trained to output single excited score
    * Textual Features + Prosodic Features trained to output joint [overall, excitement] scores

* Bootstrap Indepedent Models
  * Average(Textual Features trained to just overall, Prosodic Features trained to just overall) --> Single Overall Score 
  * Average(Textual Features trained to just excited, Prosodic Features trained to just excited) --> Single Overall excited 
  * Average(Textual Features trained to just [overall, excitement], Prosodic Features trained to just [overall, excitement]) --> Both [overall, excitement] Scores 

---

11/20
* code management - Aidan Bagley
 * Combine all scripts --> add python scripts (driver code) and requirements.txt 

* MultiModal Model
  * Experiment and check avg baseline -  Rahul and John

* Hyperparameter Tuning
  * Compile hyperparameter combinations (restrict to 'ideal' space) and then run them to display the results










  
