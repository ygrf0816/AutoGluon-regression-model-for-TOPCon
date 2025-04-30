import matplotlib.pyplot as plt
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as pl
from sklearn import metrics
import matplotlib.ticker as ticker
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
import os.path
# Related libraries

def plot_matrix(y_true, y_pred, labels_name, title=None, thresh=0.8,path='figure.png', axis_labels=None):
# Use functions from sklearn to generate confusion matrix and normalize it
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels_name, sample_weight=None)  # Generate confusion matrix 
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize
# Draw the plot, if you want to change the color style, you can change cmap=pl.get_cmap('Blues') in this part
    pl.imshow(cm, interpolation='nearest', cmap=pl.get_cmap('Blues'))
    pl.colorbar()  # Draw legend
# Image title
    if title is not None:
        pl.title(title)
# Draw coordinates
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    pl.xticks(num_local, axis_labels, rotation=45)  # Print labels on x-axis coordinates, and rotate 45 degrees
    pl.yticks(num_local, axis_labels)  # Print labels on y-axis coordinates
    pl.ylabel('Ground Truth label')
    pl.xlabel('Prediction label')

# Print percentages in the appropriate cells, use white text for values greater than thresh, and black text for less than
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                pl.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")  # If you want to change the color style, you also need to change this line
# Display
    pl.show()
    pl.savefig(path)

def draw(truth,predict,title,path,mode='valid'):
    plt.figure()
    plt.title(title)
    plt.plot(predict,'ro-', label=mode)
    plt.plot(truth, 'bx-',label='ground truth')
    plt.grid(False)
    #plt.grid(ls='--')
    plt.legend()
    plt.savefig(path, dpi=1200)

def train_model(train_loader,testnet,loss_func,optimizer,scheduler,epoch_num=50):
    # Iteratively train the model, a total of epoch rounds
    for epoch in range(epoch_num):
        train_loss = 0
        train_num = 0
        # Iterate through the loader for training data
        loop = tqdm(enumerate(train_loader), total =len(train_loader))
        for step, (b_x, b_y) in loop:
            b_x, b_y = b_x.cuda(), b_y.cuda()
            b_x=torch.flatten(b_x,start_dim=1)
            output = testnet(b_x) # MLP output on training batch

            loss = loss_func(output, b_y) # Loss function
            optimizer.zero_grad() # Initialize gradient to 0 for each iteration
            loss.backward() # Backpropagation, calculate gradient
            optimizer.step() # Use gradient for optimization
            train_loss += loss.item() * b_x.size(0)
            train_num += b_x.size(0)

            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            adjusted_lr = scheduler.get_last_lr()
            loop.set_description(f'Epoch [{epoch}/{epoch_num}]')
            loop.set_postfix(s_L1 = loss.item(),current_lr=current_lr,adjusted_lr=adjusted_lr)
            """ if (prev_loss-loss.item()<limit_loss) and (prev_loss!=loss.item()):
                print("Training stopped. Loss update is below the specified value.")
                stop_training = True
                break  # 停止训练
            prev_loss = loss.item()
        if stop_training:break """
        scheduler.step()
        return train_loss / train_num

class MLPregression(nn.Module):
    def __init__(self):
        super(MLPregression, self).__init__()
        # Hidden layers
        self.hidden1 = nn.Linear(in_features=17, out_features=8000, bias=True)
        self.hidden2 = nn.Linear(8000, 4000)
        self.hidden3 = nn.Linear(4000, 2000)
        self.hidden4 = nn.Linear(2000, 1000)
        #self.hidden5 = nn.Linear(8000, 10000)

        # Regression prediction layer
        self.predict = nn.Linear(1000, 7)

        # Batch normalization
        self.bnlayer3 = nn.BatchNorm1d(4000)
        self.bnlayer4 = nn.BatchNorm1d(8000)
        self.bnlayer2 = nn.BatchNorm1d(2000)
        self.bnlayer1 = nn.BatchNorm1d(1000)

        nn.init.kaiming_uniform_(self.hidden1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.hidden2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.hidden3.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.hidden4.weight, mode='fan_in', nonlinearity='relu')
        
    # Define network forward propagation path
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.bnlayer4(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.bnlayer3(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.bnlayer2(x))
        x = F.relu(self.hidden4(x))
        x = F.relu(self.bnlayer1(x))
        #x = F.relu(self.hidden5(x))
        #x = F.relu(self.bnlayer3(x))
        output = self.predict(x)
        # Output a one-dimensional vector
        return output

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def draw_confusion_matrix(label_true, label_pred, label_name, title="Confusion Matrix", pdf_save_path=None, dpi=100):
    """

    @param label_true: True labels, like [0,1,2,7,4,5,...]
    @param label_pred: Predicted labels, like [0,5,4,2,1,4,...]
    @param label_name: Label names, like ['cat','dog','flower',...]
    @param title: Chart title
    @param pdf_save_path: Whether to save, if yes it's the save path pdf_save_path=xxx.png | xxx.pdf | ... or other formats supported by plt.savefig
    @param dpi: Resolution for saving to file, academic papers generally require at least 300dpi
    @return:

    example：
            draw_confusion_matrix(label_true=y_gt,
                          label_pred=y_pred,
                          label_name=["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"],
                          title="Confusion Matrix on Fer2013",
                          pdf_save_path="Confusion_Matrix_on_Fer2013.png",
                          dpi=300)

    """
    #set font size
    #plt.rcParams.update({'font.size': 12})

    cm = confusion_matrix(y_true=label_true, y_pred=label_pred, normalize='true')

    plt.imshow(cm, cmap='Blues')
    plt.title(title)
    plt.xlabel("Prediction label",fontsize=13)
    plt.ylabel("Ground Truth label",fontsize=13)
    plt.yticks(range(label_name.__len__()), label_name,fontsize=13)
    plt.xticks(range(label_name.__len__()), label_name,fontsize=13, rotation=45)

    plt.tight_layout()

    cb1=plt.colorbar()
    tick_locator = ticker.MaxNLocator(nbins=5)  # Number of tick values on the colorbar
    cb1.locator = tick_locator
    cb1.set_ticks([0, 0.25,0.5, 0.75, 1])
    # cb1.ax.tick_params(labelsize=13)
    cb1.update_ticks()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # White font for diagonal, black for others
            value = float(format('%.2f' % cm[j, i]))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    # plt.show()
    if not pdf_save_path is None:
        plt.savefig(pdf_save_path, bbox_inches='tight', dpi=dpi)

def scatter_matrix(true_values,predicted_values,xlabel='Predicted',ylabel='Truth',
                   title=None,path='figure/scatter_matrix.png'):

    # Create a scatter plot
    plt.scatter(true_values, predicted_values, c=true_values, cmap='jet', s=10)
    plt.colorbar(label='N')

    # Draw a y=x line, representing perfect prediction
    plt.plot([0,1], [0,1], 'k--')

    # Set axis labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title!=None:plt.title(title)

    plt.savefig(path,dpi=1200)

class MultilabelPredictor():
    """ Tabular Predictor for predicting multiple columns in table.
        Creates multiple TabularPredictor objects which you can also use individually.
        You can access the TabularPredictor for a particular label via: `multilabel_predictor.get_predictor(label_i)`

        Parameters
        ----------
        labels : List[str]
            The ith element of this list is the column (i.e. `label`) predicted by the ith TabularPredictor stored in this object.
        path : str, default = None
            Path to directory where models and intermediate outputs should be saved.
            If unspecified, a time-stamped folder called "AutogluonModels/ag-[TIMESTAMP]" will be created in the working directory to store all models.
            Note: To call `fit()` twice and save all results of each fit, you must specify different `path` locations or don't specify `path` at all.
            Otherwise files from first `fit()` will be overwritten by second `fit()`.
            Caution: when predicting many labels, this directory may grow large as it needs to store many TabularPredictors.
        problem_types : List[str], default = None
            The ith element is the `problem_type` for the ith TabularPredictor stored in this object.
        eval_metrics : List[str], default = None
            The ith element is the `eval_metric` for the ith TabularPredictor stored in this object.
        consider_labels_correlation : bool, default = True
            Whether the predictions of multiple labels should account for label correlations or predict each label independently of the others.
            If True, the ordering of `labels` may affect resulting accuracy as each label is predicted conditional on the previous labels appearing earlier in this list (i.e. in an auto-regressive fashion).
            Set to False if during inference you may want to individually use just the ith TabularPredictor without predicting all the other labels.
        kwargs :
            Arguments passed into the initialization of each TabularPredictor.

    """
    
    multi_predictor_file = 'iv_predictor.pkl'

    def __init__(self, labels, path=None, problem_types=None, eval_metrics=None, consider_labels_correlation=True, **kwargs):
        if len(labels) < 2:
            raise ValueError("MultilabelPredictor is only intended for predicting MULTIPLE labels (columns), use TabularPredictor for predicting one label (column).")
        if (problem_types is not None) and (len(problem_types) != len(labels)):
            raise ValueError("If provided, `problem_types` must have same length as `labels`")
        if (eval_metrics is not None) and (len(eval_metrics) != len(labels)):
            raise ValueError("If provided, `eval_metrics` must have same length as `labels`")
        self.path = setup_outputdir(path, warn_if_exist=False)
        self.labels = labels
        self.consider_labels_correlation = consider_labels_correlation
        self.predictors = {}  # key = label, value = TabularPredictor or str path to the TabularPredictor for this label
        if eval_metrics is None:
            self.eval_metrics = {}
        else:
            self.eval_metrics = {labels[i] : eval_metrics[i] for i in range(len(labels))}
        problem_type = None
        eval_metric = None
        for i in range(len(labels)):
            label = labels[i]
            path_i = self.path + "Predictor_" + label
            if problem_types is not None:
                problem_type = problem_types[i]
            if eval_metrics is not None:
                eval_metric = eval_metrics[i]
            self.predictors[label] = TabularPredictor(label=label, problem_type=problem_type, eval_metric=eval_metric, path=path_i, **kwargs)

    def fit(self, train_data, tuning_data=None, **kwargs):
        """ Fits a separate TabularPredictor to predict each of the labels.

            Parameters
            ----------
            train_data, tuning_data : str or autogluon.tabular.TabularDataset or pd.DataFrame
                See documentation for `TabularPredictor.fit()`.
            kwargs :
                Arguments passed into the `fit()` call for each TabularPredictor.
        """
        if isinstance(train_data, str):
            train_data = TabularDataset(train_data)
        if tuning_data is not None and isinstance(tuning_data, str):
            tuning_data = TabularDataset(tuning_data)
        train_data_og = train_data.copy()
        if tuning_data is not None:
            tuning_data_og = tuning_data.copy()
        else:
            tuning_data_og = None
        save_metrics = len(self.eval_metrics) == 0
        for i in range(len(self.labels)):
            label = self.labels[i]
            predictor = self.get_predictor(label)
            if not self.consider_labels_correlation:
                labels_to_drop = [l for l in self.labels if l != label]
            else:
                #labels_to_drop = [self.labels[j] for j in range(i+1, len(self.labels))]
                a1 = [self.labels[j] for j in range(0, i-5)]
                a2 = [self.labels[j] for j in range(i+1, len(self.labels))]
                labels_to_drop = a1 + a2
            train_data = train_data_og.drop(labels_to_drop, axis=1)
            if tuning_data is not None:
                tuning_data = tuning_data_og.drop(labels_to_drop, axis=1)
            print(f"Fitting TabularPredictor for label: {label} ...")
            predictor.fit(train_data=train_data, tuning_data=tuning_data, **kwargs)
            self.predictors[label] = predictor.path
            if save_metrics:
                self.eval_metrics[label] = predictor.eval_metric
        self.save()

    def predict(self, data, **kwargs):
        """ Returns DataFrame with label columns containing predictions for each label.

            Parameters
            ----------
            data : str or autogluon.tabular.TabularDataset or pd.DataFrame
                Data to make predictions for. If label columns are present in this data, they will be ignored. See documentation for `TabularPredictor.predict()`.
            kwargs :
                Arguments passed into the predict() call for each TabularPredictor.
        """
        return self._predict(data, as_proba=False, **kwargs)

    def predict_proba(self, data, **kwargs):
        """ Returns dict where each key is a label and the corresponding value is the `predict_proba()` output for just that label.

            Parameters
            ----------
            data : str or autogluon.tabular.TabularDataset or pd.DataFrame
                Data to make predictions for. See documentation for `TabularPredictor.predict()` and `TabularPredictor.predict_proba()`.
            kwargs :
                Arguments passed into the `predict_proba()` call for each TabularPredictor (also passed into a `predict()` call).
        """
        return self._predict(data, as_proba=True, **kwargs)

    def evaluate(self, data, **kwargs):
        """ Returns dict where each key is a label and the corresponding value is the `evaluate()` output for just that label.

            Parameters
            ----------
            data : str or autogluon.tabular.TabularDataset or pd.DataFrame
                Data to evalate predictions of all labels for, must contain all labels as columns. See documentation for `TabularPredictor.evaluate()`.
            kwargs :
                Arguments passed into the `evaluate()` call for each TabularPredictor (also passed into the `predict()` call).
        """
        data = self._get_data(data)
        eval_dict = {}
        for label in self.labels:
            print(f"Evaluating TabularPredictor for label: {label} ...")
            predictor = self.get_predictor(label)
            eval_dict[label] = predictor.evaluate(data, **kwargs)
            if self.consider_labels_correlation:
                data[label] = predictor.predict(data, **kwargs)
        return eval_dict

    def save(self):
        """ Save MultilabelPredictor to disk. """
        for label in self.labels:
            if not isinstance(self.predictors[label], str):
                self.predictors[label] = self.predictors[label].path
        save_pkl.save(path=self.path+self.multi_predictor_file, object=self)
        print(f"MultilabelPredictor saved to disk. Load with: MultilabelPredictor.load('{self.path}')")

    @classmethod
    def load(cls, path):
        """ Load MultilabelPredictor from disk `path` previously specified when creating this MultilabelPredictor. """
        path = os.path.expanduser(path)
        if path[-1] != os.path.sep:
            path = path + os.path.sep
        return load_pkl.load(path=path+cls.multi_predictor_file)

    def get_predictor(self, label):
        """ Returns TabularPredictor which is used to predict this label. """
        predictor = self.predictors[label]
        if isinstance(predictor, str):
            return TabularPredictor.load(path=predictor)
        return predictor

    def _get_data(self, data):
        if isinstance(data, str):
            return TabularDataset(data)
        return data.copy()

    def _predict(self, data, as_proba=False, **kwargs):
        data = self._get_data(data)
        if as_proba:
            predproba_dict = {}
        for label in self.labels:
            print(f"Predicting with TabularPredictor for label: {label} ...")
            predictor = self.get_predictor(label)
            if as_proba:
                predproba_dict[label] = predictor.predict_proba(data, as_multiclass=True, **kwargs)
            data[label] = predictor.predict(data, **kwargs)
        if not as_proba:
            return data[self.labels]
        else:
            return predproba_dict
