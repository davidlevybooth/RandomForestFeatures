# RandomForestFeatures.py

Random Forest Feature Selection

Description: 
Python implimentation of the Scikit-Learn Random Forest Classifier to select features (variables) associated with catagories or treatments. 
Random forests are ensemble classifiers of classification and regression trees. 
Use Boruta feature selection to permute the random forest classifier to calculate significance of associated features.

This script has options to output:
1. A .csv of your features (variables) with ranked Random Forest importance
2. A plot (.png) of your features with ranked Random Forest importance
1. A .csv of your features with ranked Boruta importance
2. A plot of your features with ranked Boruta importance

Requirements:
Python 2.6 or higher (not Python 3): https://www.python.org/downloads/
Pandas: http://pandas.pydata.org/getpandas.html
Numpy: http://www.numpy.org/
SciKit-Learn 0.14.0 or higher: http://scikit-learn.org/stable/install.html
Boruta_py: https://github.com/danielhomola/boruta_py

Ensure SciKit-Learn is installed. If you've managed to get SciKit-Learn working, the other dependancies are probably in place. 
Download Boruta_py from github. Ensure that it either in your path, or in the same folder as RandomForestFeatures.py

Usage: 
In the terminal (or command line for weirdos who use windows): 

RandomForestFeatures.py

Required arguments: 
-i (--input_fp) : path to the input .csv file with columns for your data and one catagorical column (string)
-p (--predvar) : catagorical variable column against which features will be selected (string)

Optional arguments: 
-g (--plot) : produce a feature plot? (boolean, default = true)
-o (--output) : output folder name (string, default = rf_output)
-f (--featdepth) : number of feature to plot (integer, default = 10)
-b (--boruta) : perform Boruta feature selection (boolean, IMPORTANT: default is FALSE)

Example: 

RandomForestFeatures.py -i dataToClassify.csv -p Catagory -o output_folder -f 20 -b true

Advanced:
Aditional variables related to the Random Forest parameters and Boruta feature selection method can be altered in the script

Default Random Forest parameters: 
rf_params = {'n_estimators': 1000, 'max_depth': 10, 'min_samples_split': 1}

Default Boruta method: 
feat_selector = BorutaPy(rf, multi_corr_method='hommel', n_estimators='auto', max_iter = 100, verbose=1)
