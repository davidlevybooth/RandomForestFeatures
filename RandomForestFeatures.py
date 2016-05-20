import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

from boruta_py import BorutaPy
from itertools import compress

import argparse
import os

###############################################################################

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_fp", required=True,
	help="path to the input OTU file", type=str)
ap.add_argument("-p", "--predvar", required=True,
	help="names of variables for catagorization", type=str)
ap.add_argument("-g", "--plot",
	help="save feature plot", type=bool, default=True)
ap.add_argument("-o", "--output", required=False,
	help="output directory", type=str, default="rf_ouput")
ap.add_argument("-f", "--featdepth", required=False,
	help="depth of features to plot", type=int, default=10)
ap.add_argument("-b", "--boruta", required=False,
	help="depth of features to plot", type=bool, default=False)
args = vars(ap.parse_args())

#assign arguments
file_path = args["input_fp"]
predvar = args["predvar"]
plot_bool = args["plot"]
output_path = args["output"]
feature_depth = args["featdepth"]
boruta = args["boruta"]

#ensure output directory exists
if not os.path.exists(output_path):
    os.makedirs(output_path)

###############################################################################


# Load data
full_data = pd.read_csv(file_path)
predvar_data = full_data[predvar].values.tolist() 
predictors = full_data.drop(predvar, axis=1)

# data values in lists for feature selection
feat_X = predictors.values
X_names = list(predictors.columns)

# turn catagory variable into numeric variable
le = LabelEncoder() # sklearn encoder allows variables
le.fit(list(set(predvar_data))) # get only unique catagory values
predvar_numeric = le.transform(predvar_data)

###############################################################################

#Define Random Forest parameters
rf_params = {'n_estimators': 1000, 'max_depth': 10, 'min_samples_split': 1}
rf = RandomForestClassifier(**rf_params) # **rf_params

# ###############################################################################

# produce training and test data to determine classification accuracy
X, y = shuffle(feat_X, predvar_numeric, random_state=13) # feat_X <------- new_X
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# ###############################################################################

# Fit RF classification model for MSE
rf.fit(X_train, y_train)

rf_mse = mean_squared_error(y_test, rf.predict(X_test))
print("RF MSE: %.4f" % rf_mse)

# # ###############################################################################

# Fit full RF classification model
rf.fit(X, predvar_numeric)

# #################################################################################

if plot_bool: 
	# Plot RF feature importance
	plt.figure()

	feature_importance = rf.feature_importances_
	# make importances relative to max importance
	# feature_importance = 100.0 * (feature_importance / feature_importance.max())
	sorted_idx = np.argsort(feature_importance) # x axis data
	plot_fi =  feature_importance[sorted_idx]
	plot_fi = plot_fi[-feature_depth:]

	# Assign plot attributes
	sorted_x_vars = [ X_names[i] for i in list(sorted_idx)] # y labels
	sorted_x_vars = sorted_x_vars[-feature_depth:]

	pos = np.arange(sorted_idx[-feature_depth:].shape[0]) + .5
	# plt.subplot(1, 2, 1)
	plt.barh(pos, plot_fi, align='center')
	plt.yticks(pos, sorted_x_vars)
	plt.xlabel('Relative Importance')
	plt.title("Random Forest MSE: %.4f" % rf_mse)
	
	#save file 
	plt.savefig(output_path + "/RF_" + predvar + "_" + str(feature_depth) + "_feature_importance.png", bbox_inches='tight')

#################################################################################

	# Save first # of OTUS
	reduced_X = predictors[sorted_x_vars]
	reduced_X.columns = sorted_x_vars

	# np.savetxt(output_path + "/RF_boruta_output_new_X.csv", new_X, delimiter=",") # needs to have col otu names and row names
	reduced_X.to_csv(output_path + "/RF_" + predvar + "_" + str(feature_depth) + "_output.csv", index=False)

#################################################################################

# Boruta feature selection 
if boruta: 
	# define Boruta feature selection method
	feat_selector = BorutaPy(rf, multi_corr_method='hommel', n_estimators='auto', max_iter = 100, verbose=1) #hommel works well

	# find all relevant features
	feat_selector.fit(feat_X, predvar_numeric)

	# check selected features
	feat_bools = list(feat_selector.support_)
	feat_strong_list = list(compress(X_names, feat_bools))
	feat_weak_list = list(compress(X_names, list(feat_selector.support_weak_)))

	feat_list = feat_strong_list + feat_weak_list
	
	new_X = predictors[feat_list]
	new_X.columns = feat_list

	# call transform() on X to filter it down to selected features
	X_filtered = feat_selector.transform(feat_X) # vestigial list of only strong selectors
	#save filtered features
	# np.savetxt(output_path + "/RF_boruta_output_new_X.csv", new_X, delimiter=",") # needs to have col otu names and row names
	new_X.to_csv(output_path + "/RF_boruta_" + predvar + "_" + "output.csv", index=False)

	###############################################################################

	# produce training and test data to determine classification accuracy
	X, y = shuffle(new_X, predvar_numeric, random_state=13) # feat_X <------- new_X
	X = X.astype(np.float32)
	offset = int(X.shape[0] * 0.9)
	X_train, y_train = X[:offset], y[:offset]
	X_test, y_test = X[offset:], y[offset:]

	X_vars = list(predictors.columns) # this is used when ploting feature importance

	# ###############################################################################

	# Fit RF classification model for MSE
	rf.fit(X_train, y_train)

	rf_mse = mean_squared_error(y_test, rf.predict(X_test))
	print("Boruta RF MSE: %.4f" % rf_mse)

	# # ###############################################################################

	# Fit full RF classification model
	rf.fit(new_X, predvar_numeric)

	# #################################################################################

	if plot_bool: 
		plt.figure()
		# Plot RF feature importance
		depth = len(feat_list)

		feature_importance = rf.feature_importances_
		# make importances relative to max importance
		# feature_importance = 100.0 * (feature_importance / feature_importance.max())
		sorted_idx = np.argsort(feature_importance) # x axis data
		plot_fi =  feature_importance[sorted_idx]
		# plot_fi = plot_fi[-depth:]

		# Assign plot attributes
		sorted_x_vars = [ X_names[i] for i in list(sorted_idx)] # y labels
		# sorted_x_vars = sorted_x_vars[-depth:]

		# pos = np.arange(sorted_idx[-depth:].shape[0]) + .5
		pos = np.arange(sorted_idx.shape[0]) + .5
		# plt.subplot(1, 2, 1)
		plt.barh(pos, plot_fi, align='center')
		plt.yticks(pos, sorted_x_vars)
		plt.xlabel('Relative Importance')
		plt.title("Random Forest MSE: %.4f" % rf_mse)
		
		#save file 
		plt.savefig(output_path + "/RF_boruta_" + predvar + "_" + "_feature_importance.png", bbox_inches='tight')
