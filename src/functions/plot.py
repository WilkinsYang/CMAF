import matplotlib.pyplot as plt
import numpy as np
import itertools
import seaborn as sns
import matplotlib.ticker as mtick

def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('train history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

# confusion matrix
def plot_confusion_matrix(cm, title, cmap=plt.cm.Oranges):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[1])
    plt.xticks(tick_marks)
    ax = plt.gca()
    ax.set_xticklabels((ax.get_xticks()).astype(str))
    plt.yticks(tick_marks)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def percentage(cm):
    fig, ax = plt.subplots(figsize=(5,5))
    ax = plt.gca()
    tick_marks = np.arange(cm.shape[1])
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=tick_marks, yticklabels=tick_marks,cbar=True, cbar_kws={'format':mtick.PercentFormatter()},cmap="Blues")
    plt.tight_layout()
    plt.yticks(rotation=0) 
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion matrix')
    plt.show()


