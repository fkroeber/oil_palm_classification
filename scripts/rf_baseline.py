import numpy as np
import pandas as pd
import rasterio
from imblearn.over_sampling import RandomOverSampler
from itertools import product
from skimage.color import rgb2gray, label2rgb
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from skimage.feature import canny, SIFT
from skimage.segmentation import felzenszwalb
from skimage.transform import probabilistic_hough_line
from tqdm import tqdm
import scripts.utils as utils


class OilPalmRF:
    def __init__(self, img_paths, labels):
        """Provide absolute paths to images (X) and labels (y)"""
        self.img_paths = img_paths
        self.labels = labels

    def feat_engineering(self, save_path=None):
        """Perform feature engineering on a set of images
        1. SIFT & HOG features
        2. Hough transform & line features
        3. Segmentation & spectral features"""
        feat_eng_res = {}
        for img_path in tqdm(
            self.img_paths,
            smoothing=0.001,
            desc="Feature engineering for images",
        ):
            feat_descrpts = {}
            # read & preprocess image
            src = rasterio.open(img_path)
            img = src.read().transpose(1, 2, 0)
            img_grey = rgb2gray(img)
            # SIFT feature extraction & keypoint description via HOG
            thres_contrast = 0.0125
            sift = SIFT(c_dog=thres_contrast)
            while True:
                try:
                    sift.detect_and_extract(img_grey)
                    break
                except RuntimeError:
                    thres_contrast *= 0.9
                    sift = SIFT(c_dog=thres_contrast)
            feat_descrpts["sift"] = sift.descriptors
            # hough transform feature extraction & description
            edges = canny(img_grey, 1)
            lines = probabilistic_hough_line(edges, line_length=33, line_gap=1)
            if len(lines):
                dists = [
                    np.sqrt((x[0][0] - x[1][0]) ** 2 + (x[0][1] - x[1][1]) ** 2)
                    for x in lines
                ]
            line_len_avg = np.mean(dists) if len(lines) else 0
            feat_descrpts["line"] = np.array((len(lines), line_len_avg))
            # segmentation-based feature extraction & description
            segments = felzenszwalb(img_grey, scale=200, min_size=50)
            img_avg = label2rgb(segments, img, kind="avg", bg_label=-1)
            uniq_idxs, sizes = np.unique(segments, return_counts=True)
            mean_rgbs = {}
            for value in uniq_idxs:
                mask = np.where(segments == value, 1, 0)
                selected_values = img_avg[mask == 1]
                mean = np.mean(selected_values, axis=0)
                mean_rgbs[value] = mean
            mean_rgbs = np.vstack(list(mean_rgbs.values()))
            feat_descrpts["segments"] = np.hstack((sizes.reshape(-1, 1), mean_rgbs))
            feat_eng_res[img_path] = feat_descrpts
        self.feat_eng_res = feat_eng_res
        if save_path:
            utils.compress_pickle(save_path, self.feat_eng_res)

    def train_eval_model(self, save_path):
        """Nested cross-validation for hyperparameter tuning & model evaluation
        1. Grid search in cross-validation manner to tune hyperparameters for feature engineering
        2. Test accuracy assessment via cross-validation"""
        # define hyperparameters to be tested
        hyperparams = {
            "BoW_k_sift": [4, 6, 8, 10, 20],
            "BoW_k_segments": [4, 6, 8, 10, 20],
        }
        hyper_combis = list(product(*list(hyperparams.values())))
        hyper_combis = [dict(zip(hyperparams.keys(), combi)) for combi in hyper_combis]
        mod_res = []
        # perform grid search
        for hyperparam_config in tqdm(
            hyper_combis,
            desc="Grid search hyperparameters & cross-validation test accuracies",
        ):
            # bag of visual words (vw) to summarise sift features
            kmeans = KMeans(
                n_clusters=hyperparam_config["BoW_k_sift"],
                random_state=42,
                n_init="auto",
            )
            kmeans = kmeans.fit(
                np.vstack([x["sift"] for x in self.feat_eng_res.values()])
            )
            vw_preds = [kmeans.predict(x["sift"]) for x in self.feat_eng_res.values()]
            vw_freq = [
                np.histogram(x, bins=hyperparam_config["BoW_k_sift"])[0]
                for x in vw_preds
            ]
            vw_sift_freq = (
                np.vstack(vw_freq) / np.vstack(vw_freq).sum(axis=1).reshape(-1, 1)
            ).round(2)
            # bag of visual words (vw) to summarise segmentation features
            kmeans = KMeans(
                n_clusters=hyperparam_config["BoW_k_segments"],
                random_state=42,
                n_init="auto",
            )
            kmeans = kmeans.fit(
                np.vstack([x["segments"] for x in self.feat_eng_res.values()])
            )
            vw_preds = [
                kmeans.predict(x["segments"]) for x in self.feat_eng_res.values()
            ]
            vw_freq = [
                np.histogram(x, bins=hyperparam_config["BoW_k_segments"])[0]
                for x in vw_preds
            ]
            vw_segments_freq = np.vstack(vw_freq)
            # construct final feature engineered df
            feat_df = np.hstack(
                (
                    [x["line"] for x in self.feat_eng_res.values()],
                    vw_sift_freq,
                    vw_segments_freq,
                )
            )
            # perform training & evaluation for multiple splits
            # cross-validation for test accuracies
            mod_res_ = []
            for seed in np.arange(0, 5):
                accs = []
                # divide data into train-val-test (60-20-20)
                X_, X_test, y_, y_test = train_test_split(
                    feat_df,
                    np.array(self.labels),
                    stratify=np.array(self.labels),
                    test_size=0.20,
                    random_state=seed,
                )
                # cross-validation for hyperparameter tuning
                skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
                self.feat_df = feat_df
                self.X_ = X_
                self.y_ = y_
                for train, val in skf.split(X_, y_):
                    X_train, y_train = X_[train], y_[train]
                    X_val, y_val = X_[val], y_[val]
                    ros = RandomOverSampler(random_state=42)
                    X_train, y_train = ros.fit_resample(X_train, y_train)
                    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
                    mod = rf.fit(X_train, y_train)
                    accs.append(mod.score(X_val, y_val))
                accs = pd.DataFrame(accs).T
                accs.columns = [f"split_{x}" for x in np.arange(4)]
                accs["val_acc_mean"] = accs.mean(axis=1)
                accs["val_acc_std"] = accs.std(axis=1)
                # assess test performance
                rf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
                rf.fit(X_, y_.flatten())
                accs["test_acc"] = rf.score(X_test, y_test)
                accs.insert(0, "test_split", seed)
                mod_res_.append(accs)
            mod_res_ = pd.concat(mod_res_).reset_index(drop=True)
            mod_res_ = mod_res_.assign(**hyperparam_config)
            cols = list(hyperparam_config.keys()) + list(mod_res_.columns)[:-2]
            mod_res.append(mod_res_.reindex(columns=cols))
        self.mod_res = pd.concat(mod_res).reset_index(drop=True)
        if save_path:
            utils.compress_pickle(save_path, self.mod_res)
