import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance
import pickle
from func import draw_confusion_matrix
import time
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap
import random
# Global settings
from proplot import rc
# Set font uniformly
# rc["font.family"] = "Times New Roman"
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['text.latex.preamble'] = r'''
    \usepackage{mathptmx}  % vtilizes the Times New Roman font in LaTeX
'''
# Set font size for axis tick labels uniformly
rc['tick.labelsize'] = 17
# Set font size for xy axis names uniformly
rc["axes.labelsize"] = 17
# Set font size for colorbar labels uniformly
#rc['colorbar.labelsize'] = 24
# Set font weight for axis tick labels uniformly
rc["axes.labelweight"] = "regular"
# Set font weight for xy axis names uniformly
rc["tick.labelweight"] = "regular"
rc["grid"]=False

# 1. Prepare data
# Training labels
# is_train = False
is_train = True

#======================
# Jsc: 4 categories; Voc: 3 categories; FF: 4 categories; Eff: 5 categories
#======================
target = 'Eff'
filename = f'dataset/TOPCon_high{target}.csv'
# filename = 'dataset/Three_classdata.csv'
class_num = 5 # Modify according to the label

# class_names=["Jsc 1", "Jsc 2", "Jsc 3", "Jsc 4"]
# class_names=["Voc 1", "Voc 2", "Voc 3"]
# class_names=["FF 1", "FF 2", "FF 3","FF 4"]
class_names=["Eff 1", "Eff 2", "Eff 3", "Eff 4", "Eff 5"]
# class_names=["Eff 1", "Eff 2", "Eff 3"]

model_file = f'Models//final//lgb_model_high{target}.pkl'
# model_file = 'Models//final//lgb_model_3class.pkl'
# The feature order here needs to be consistent with the variable order in importance plot and other charts
# HighEff LGBM importance plot
# features = ['t_SiO2', 'Nt_rear', 't_polySi_rear', 'Sv_SiOx_Poly', 
#             't_Si', 'Sv_top', 'Na_top', 'Sv_Si_SiOx', 'Nd_rear', 
#             'Nt_front', 'rear_junc', 'Rs', 'front_junc']

# label replace, change into latex format
# LowEff & BadEff importance plot
features =  ['Nt_polySi_top', 't_polySi_rear_P', 'Nt_polySi_rear', 'resist_rear', 
            'Si_thk', 'Dit top', 'Dit SiOx-Poly', 'rear_junc', 'Nd_rear', 
            'Nd_top', 'front_junc', 'Dit Si-SiOx', 't_SiO2']

# HighEff ln version          
# latex_features = [r'$\mathit{t_\mathrm{SiO_x}}$',r'$\mathit{ln(N_\mathrm{t~rear})}$',r'$\mathit{t_\mathrm{PolySi}}$',
#             r'$\mathit{ln(S_\mathrm{v~SiO_x-PolySi})}$',r'$\mathit{t_\mathrm{c-Si}}$',r'$\mathit{ln(S_\mathrm{v-front})}$',
#             r'$\mathit{ln(N_\mathrm{a~front})}$',r'$\mathit{ln(S_\mathrm{v~Si-SiOx})}$',r'$\mathit{ln(N_\mathrm{d~PolySi})}$',
#             r'$\mathit{ln(N_\mathrm{t~front})}$',r'$\mathit{d_\mathrm{rear}}$',r'$\mathit{R_\mathrm{s}}$',r'$\mathit{d_\mathrm{front}}$']

# HighEff w/o log version
# latex_features = [r'$\mathit{t_\mathrm{SiO_x}}$',r'$\mathit{N_\mathrm{t~rear}}$',r'$\mathit{t_\mathrm{PolySi}}$',
#             r'$\mathit{S_\mathrm{v~SiO_x-PolySi}}$',r'$\mathit{t_\mathrm{c-Si}}$',r'$\mathit{S_\mathrm{v-front}}$',
#             r'$\mathit{N_\mathrm{a~front}}$',r'$\mathit{S_\mathrm{v~Si-SiOx}}$',r'$\mathit{N_\mathrm{d~PolySi}}$',
#             r'$\mathit{N_\mathrm{t~front}}$',r'$\mathit{d_\mathrm{rear}}$',r'$\mathit{R_\mathrm{s}}$',r'$\mathit{d_\mathrm{front}}$']

# LowEff & BadEff importance plot
latex_features = [r'$\mathit{N_\mathrm{t~front}}$',r'$\mathit{t_\mathrm{PolySi}}$',r'$\mathit{N_\mathrm{t~rear}}$',
            r'$\mathit{R_\mathrm{s}}$',r'$\mathit{t_\mathrm{cSi}}$',r'$\mathit{D_\mathrm{it-front}}$',
            r'$\mathit{D_\mathrm{it~SiO_x-PolySi}}$',r'$\mathit{d_\mathrm{rear}}$',r'$\mathit{N_\mathrm{d~PolySi}}$',
            r'$\mathit{N_\mathrm{a~front}}$',r'$\mathit{d_\mathrm{front}}$',r'$\mathit{D_\mathrm{it~Si-SiO_x}}$',r'$\mathit{t_\mathrm{SiO_x}}$']

data_length = 13

train_test = pd.read_csv(filename)
a = train_test
train_test=train_test.drop(columns=[f'{target}'])

x = train_test.iloc[:,0:data_length]
column = x.columns


x_display = np.array(x)
# Convert exponential form values in x_display to logarithmic form
subset_to_transform = x_display[:, 6:13]
subset_transformed = np.log10(subset_to_transform)
x_display[:, 6:13] = subset_transformed


x_original=pd.DataFrame(x_display)
x_original.columns = column
scaler = StandardScaler().fit(x)
x=pd.DataFrame(scaler.transform(x))
x.columns=column
# The training model requires that the class column data must start from 0, but the current dataset class column starts from 1
# manually change the class column data to start from 0
y = train_test['Class'] - 1


# 2. Split training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1,shuffle=True)

# 3. Create LightGBM dataset
lgb_train = lgb.Dataset(x, label=y)
lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

# 4. Set parameters
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': class_num,  
    'metric': {'multi_logloss'},
    'num_leaves': 31,
    'learning_rate': 0.0005,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1
}

###
from lightgbm import log_evaluation, early_stopping
callbacks = [log_evaluation(period=100), early_stopping(stopping_rounds=30)]

# 5. Train model
if is_train == True:
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=300,
                    valid_sets=(lgb_train, lgb_eval),
                     ###
                    callbacks=callbacks
                    ###
                    )
    pickle.dump(gbm,open(model_file,'wb'))
    # Export classification model structure
    lgb.plot_tree(gbm,precision=3,show_info=['split_gain','data_percentage','internal_value'],
                  tree_index=1,orientation='vertical')
    plt.tight_layout()
    plt.savefig('figure/lgb_structure_3C_v.png',dpi=1000)
    lgb.plot_tree(gbm,precision=3,show_info=['split_gain','data_percentage','internal_value'],
                  tree_index=1,orientation='horizontal')
    plt.tight_layout()
    plt.savefig('figure/lgb_structure_3C_h.png',dpi=1000)
    
else:
    print('load model')
    gbm = pickle.load(open(model_file,'rb'),encoding='bytes')

# 6. Model prediction
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred = np.argmax(y_pred, axis=1)  # Convert probabilities to categories
y_pred_df = pd.DataFrame(y_pred)

# Save prediction results
result = pd.concat([X_test,y_test],axis=1)
pd.concat([result,y_pred_df],axis=1).to_csv('output/result_multi_LGBM.csv',index=None)

# 7. Evaluate model
if is_train==True:
    accuracy = accuracy_score(y_test, y_pred)
    # f = open(('output/log_imp_LGBM_'+str(time.time())+'.txt'),'w')
    f = open(('1218figure/log_imp_LGBM_3C.txt'),'w')
    print("Accuracy:", accuracy,file=f)
    print(classification_report(y_test, y_pred),file=f)
    draw_confusion_matrix(label_true=y_test,label_pred=y_pred,
                            label_name=class_names,
                            title="Confusion Matrix of Multi Classes by LGBM",
                            pdf_save_path=f"1218figure/auto/Confusion_Matrix_by_lgbm_3C.png",dpi=300)
                            # pdf_save_path=f"1218figure/auto/Confusion_Matrix_by_lgbm_{target}.png",dpi=300)

# Use shap library to calculate feature importance
explainer = shap.TreeExplainer(gbm)
print('Calculate shap values')
shap_values = explainer.shap_values(x)

shap_de_values = np.array(shap_values)

print('Start Plotting')

# Beeswarm plot
plt.rc('font', family='Times New Roman')
shap.initjs()
u=np.array(a['Eff'])
print('Beeswarm plot start')
for i in range(class_num):
    #colorbar=False
    plt.figure()
    plt.cla()
    plt.grid(False)
    shap_values_class = shap_de_values[i,:,:]
    # shap_values_class = shap.Explanation(values=shap_values_class, 
    #                                  base_values=explainer.expected_value[i], 
    #                                  data=x, 
    #                                  feature_names=features)
    shap_values_class = shap.Explanation(values=shap_values_class, 
                                     base_values=explainer.expected_value[i], 
                                     data=x)
    # print(shap_values_class.shape)
    # print(x.shape)
    #if i == 0: colorbar=True
    shap.plots.beeswarm(shap_values=shap_values_class,
                        max_display=data_length,show=False,color_bar=True)
    # Get the current chart object
    ax = plt.gca()
    # Get all tick labels on the y-axis (these should be the feature names of the model)
    original_labels = [item.get_text() for item in ax.get_yticklabels()]
    #original_xlabels= [item.get_text() for item in ax.get_xticklabels()]
    # Use mapping to set the y-axis tick labels to LaTeX format labels
    label_map = dict(zip(features, latex_features))
    # Only replace when the original label is in the mapping
    y1_label = ax.get_yticklabels() 
    [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
    ax.set_yticklabels([label_map.get(label, label) for label in original_labels],fontsize=17)
    # ax.set_xticklabels([label for label in original_xlabels],fontsize=14)
    ax.set_xlabel(f'Class {i} SHAP Values',fontsize=17)
    plt.tight_layout()
    import os
    if not os.path.exists('./figure/beeswarm'):
        # if not exist, create the folder
        os.makedirs('./figure/beeswarm')
    plt.savefig('figure/beeswarm/beeswarm_'+str(i)+'.png',dpi=1000)
    plt.cla()
    plt.close('all')

# draw importance&SHAP plot
# run the code below should in SHAP==0.40.0 circumstance
plt.cla()
plt.grid(False)
print('importance plot start')
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
# Create color list and category name list
class_names = ['Class 1', 'Class 2','Class 3'
,'Class 4'
]
colors = ['#8ECFC9', '#FFBE7A', '#fa7F6F'
,'#82B0D2'
]
cmap = mcolors.ListedColormap(colors)
shap.summary_plot(shap_values=shap_values,features=x_original,show=False,plot_type='bar')
ax = plt.gca()
#===================If there's an error, you can comment out the code below==================
# Get all tick labels on the y-axis (these should be the feature names of the model)
original_labels = [item.get_text() for item in ax.get_yticklabels()]
print(original_labels)
# Use mapping to set the y-axis tick labels to LaTeX format labels
label_map = dict(zip(features, latex_features))
# Only replace when the original label is in the mapping
ax.set_yticklabels([label_map.get(label, label) for label in original_labels],fontsize=19)
plt.xticks(fontsize=18)
plt.legend(prop = {'size':16},frameon=False)
#========================================================================
ax.set_xlabel('Mean(|SHAP|) Values',fontsize=19)
# plt.tight_layout()
plt.savefig('1218figure//summary_plot_3C.png',dpi=1000)
plt.savefig(f'1218figure//summary_plot_{target}.png',dpi=1000)
 
# dependence plot
print('dependence plot start')
count=1
import os

# here we only plot the class 4（High Efficiency）
# you can change it to other classes, or sweep all classes
# for i in range(class_num):
i=4
for idx, (name1, latex_name1) in enumerate(zip(features, latex_features)):
    for name2 in features:
        plt.close('all')
        plt.figure()
        plt.cla()
        plt.grid(False)
        shap.dependence_plot(name1, shap_de_values[i,:,:], x_original, 
                                display_features=x_original,
                                interaction_index=name2,
                                show=False,colorname=latex_features[features.index(name2)],dot_size=50)

        # Get current axes
        ax = plt.gca()
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_color('k')
        ax.spines['top'].set_color('k')
        # Set x-axis label to the corresponding element in new_features
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        ax.set_xlabel(latex_name1,fontsize=16)
        # Set y-axis label to "SHAP values"
        ax.set_ylabel('SHAP Values',fontsize=16)
        plt.show()

        plt.tight_layout()
        plt.axhline(y=0,ls=":",c="black")# Add horizontal line
        
        output_dir = 'figure/dependence/class '+str(i)+'/'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_dir+
                    name1+'_'+name2+'.png',dpi=1000)
        #plt.close('all')
        print("The Class "+str(i)+' '+str(count)+" figure has been ploted")
        count=count+1#
    count=1
i=i+1    
