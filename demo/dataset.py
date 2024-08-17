import torch
import pandas as pd
import numpy as np
import nibabel as nib
import os
import fnmatch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import Dataset
import torchio as tio
import warnings
warnings.filterwarnings('ignore')

class PreprocessTransform:
    def __init__(self, new_shape):
        self.new_shape = new_shape  # New shape as a tuple (D, H, W)

    def __call__(self, img):
        # Resize the image to new_shape
        resize_transform = tio.Resize(self.new_shape)
        img_resized = resize_transform(img)
        
        # Normalize the resized image
        img_normalized = (img_resized - img_resized.mean()) / img_resized.std()
        return img_normalized

class MultimodalDataset(Dataset):
    def __init__(self, csv_file, img_folder, genetic_folder_path, transform=None, imageuids=None):
        self.tabular_frame = pd.read_csv(csv_file)
        self.img_folder = img_folder
        self.transform = transform
        self.genetic_folder_path = genetic_folder_path
        
        if imageuids is not None:
            self.tabular_frame = self.tabular_frame[self.tabular_frame['IMAGEUID'].isin(imageuids)]
        
        self.features, self.target, self.feature_size, self.tabular_row = self.preprocess_tabular_data()

    def preprocess_tabular_data(self):
        one_hot_mapping = {
            'CN': [1, 0, 0],
            'MCI': [0, 1, 0],
            'Dementia': [0, 0, 1]
        }
        
        features = self.tabular_frame.drop(columns=['IMAGEUID', 'IMAGEUID_bl', 'EXAMDATE_bl'])
        tabular_row = features.drop(columns='DX')
        
        target = features['DX'].map(one_hot_mapping)
        
        target_column = 'DX'
        feature_columns = [col for col in features.columns if col != target_column]
        
        categorical_columns = ['PTID', 'DX_bl', 'PTGENDER', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'APOE4', 'FSVERSION', 'FLDSTRENG', 'FSVERSION_bl', 'FLDSTRENG_bl']
        
        for col in categorical_columns:
            if col in feature_columns:
                features[col] = LabelEncoder().fit_transform(features[col])
        
        numerical_columns = list(set(feature_columns) - set(categorical_columns))
        scaler = StandardScaler()
        features[numerical_columns] = scaler.fit_transform(features[numerical_columns])
        
        X = features.drop(columns=[target_column])
        y = target.tolist()
        
        features_t = torch.tensor(X.values, dtype=torch.float32)
        labels = torch.tensor(y, dtype=torch.float32)
        
        feature_size = features_t.shape[1]
        
        return features_t, labels, feature_size, tabular_row

    def preprocess_genetic_data(self, df):
        nucleotide_to_number = {'A': 1, 'G': 2, 'C': 3, 'T': 4}
        for column in df.columns:
            df[column] = df[column].apply(lambda x: nucleotide_to_number[x] if x in nucleotide_to_number else np.random.randint(1, 5))
        
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_features, columns=df.columns)
        return scaled_df

    def load_genetic_data(self, ptid):
        genetic_file_path = os.path.join(self.genetic_folder_path, f"{ptid}.csv")
        genetic_df = pd.read_csv(genetic_file_path)
        scaled_gen_df = self.preprocess_genetic_data(genetic_df)
        genetic_data = torch.tensor(scaled_gen_df.values, dtype=torch.float32)
        
        return genetic_data

    def __len__(self):
        return len(self.tabular_frame)

    def __getitem__(self, idx):
        tabular_data = self.features[idx]
        ptid = self.tabular_frame.iloc[idx]['PTID']
        imageuid = self.tabular_frame.iloc[idx]['IMAGEUID']
        img_path = self.find_image_file(ptid, imageuid)
        
        if img_path:
            img = tio.ScalarImage(img_path)
            if self.transform:
                img_data = self.transform(img.data)
            else:
                img_data = img.data
            img_data = torch.tensor(img_data, dtype=torch.float32)
        else:
            img_data = torch.zeros(1, 64, 128, 128)
        
        genetic_data = self.load_genetic_data(ptid)
        
        return {
            'tabular_data': tabular_data,
            'image_data': img_data,
            'genetic_data': genetic_data,
            'label': self.target[idx],
            'img_path': img_path,
            'genetic_path': os.path.join(self.genetic_folder_path, f"{ptid}.csv"),
            'tabular_row': self.tabular_frame.iloc[idx].drop(columns=['DX']).to_string()
        }

    def find_image_file(self, ptid, imageuid):
        pattern = f'ADNI_{ptid}_*_*_I{int(imageuid)}.nii'
        for file in os.listdir(self.img_folder):
            if fnmatch.fnmatch(file, pattern):
                return os.path.join(self.img_folder, file)
        return None

# Example of using the dataset
# dataset = MultimodalDataset(csv_file='path_to_csv', img_folder='path_to_img_folder', genetic_folder_path='path_to_genetic_folder', transform=PreprocessTransform((128, 128, 128)))
# print(dataset[0])

