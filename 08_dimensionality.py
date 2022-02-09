import sys
import sklearn
import numpy as np
import os

np.random.seed(42)

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "dim_reduction"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

import warnings
warnings.filterwarnings(action="ignore", message="^internal gelsd")



import pandas as pd

def load_MHL_data():
    csv_path = os.path.join("E:/Stoleriu/C/special/3d/res/2019/Preisach-ML", "Preisach-MHL-ML-Hc0-Hcs-His-a.dat")
    return pd.read_csv(csv_path)

MHL = load_MHL_data()

print(MHL.ndim)

from sklearn.decomposition import PCA
#from sklearn.datasets import fetch_openml
#mnist = fetch_openml('mnist_784', version=1)
#mnist.target = mnist.target.astype(np.uint8)

from sklearn.model_selection import train_test_split

#X = mnist["data"]
X = MHL.iloc[:, :101]
#y = mnist["target"]
y = MHL.iloc[:, 101:]

#print(MHL)
#print(X)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y)

pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

plt.figure(figsize=(6,4))
plt.plot(cumsum, linewidth=3)
plt.axis([0, 100, 0, 1.05])
plt.xlabel("Dimensions")
plt.ylabel("Explained Variance")
plt.plot([d, d], [0, 0.95], "k:")
plt.plot([0, d], [0.95, 0.95], "k:")
plt.plot(d, 0.95, "ko")
#
plt.grid(True)
plt.show()
