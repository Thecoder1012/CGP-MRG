import os
import fnmatch
import warnings
import torch
import pandas as pd
import numpy as np
import nibabel as nib
import torchio as tio
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Suppress warnings
warnings.filterwarnings('ignore')

# Preprocessing class for image transformations
class PreprocessTransform:
    def __init__(self, new_shape):
        """
        Initialize the transform with the desired output shape.
        
        Args:
            new_shape (tuple): Target shape for the image as (Depth, Height, Width).
        """
        self.new_shape = new_shape

    def __call__(self, img):
        """
        Apply the preprocessing transform to the image data.

        Args:
            img (torch.Tensor): Image data to be transformed.

        Returns:
            torch.Tensor: Transformed image data.
        """
        resize_transform = tio.Resize(self.new_shape)
        img_resized = resize_transform(img)
        img_normalized = (img_resized - img_resized.mean()) / img_resized.std()
        return img_normalized

# Custom Dataset class for multimodal data
class MultimodalDataset(Dataset):
    def __init__(self, csv_file, img_folder, genetic_folder_path, transform=None, imageuids=None):
        """
        Initialize the dataset with tabular, image, and genetic data.

        Args:
            csv_file (str): Path to the CSV file containing tabular data.
            img_folder (str): Path to the folder containing image files.
            genetic_folder_path (str): Path to the folder containing genetic data files.
            transform (callable, optional): Optional transform to be applied to the image data.
            imageuids (list, optional): List of IMAGEUIDs to filter the dataset.
        """
        self.tabular_frame = pd.read_csv(csv_file)
        self.img_folder = img_folder
        self.transform = transform
        self.genetic_folder_path = genetic_folder_path
        
        if imageuids is not None:
            self.tabular_frame = self.tabular_frame[self.tabular_frame['IMAGEUID'].isin(imageuids)]
        
        self.features, self.target, self.feature_size, self.tabular_row = self.preprocess_tabular_data()

    def preprocess_tabular_data(self):
        """
        Preprocess the tabular data by encoding categorical features, normalizing numerical features, 
        and preparing labels.

        Returns:
            torch.Tensor: Preprocessed features.
            torch.Tensor: One-hot encoded labels.
            int: Number of features.
            pd.DataFrame: Tabular data rows without the 'DX' column.
        """
        one_hot_mapping = {'CN': [1, 0, 0], 'MCI': [0, 1, 0], 'Dementia': [0, 0, 1]}
        features = self.tabular_frame.drop(columns=['IMAGEUID', 'IMAGEUID_bl', 'EXAMDATE_bl'])
        tabular_row = features.drop(columns='DX')
        target = features['DX'].map(one_hot_mapping)

        categorical_columns = ['PTID', 'DX_bl', 'PTGENDER', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'APOE4',
                               'FSVERSION', 'FLDSTRENG', 'FSVERSION_bl', 'FLDSTRENG_bl']
        for col in categorical_columns:
            if col in features.columns:
                features[col] = LabelEncoder().fit_transform(features[col])

        numerical_columns = list(set(features.columns) - set(categorical_columns) - {'DX'})
        scaler = StandardScaler()
        features[numerical_columns] = scaler.fit_transform(features[numerical_columns])

        features_t = torch.tensor(features.drop(columns=['DX']).values, dtype=torch.float32)
        labels = torch.tensor(target.tolist(), dtype=torch.float32)
        
        return features_t, labels, features_t.shape[1], tabular_row

    def preprocess_genetic_data(self, df):
        """
        Preprocess genetic data by converting nucleotide bases to numerical values and scaling the features.

        Args:
            df (pd.DataFrame): DataFrame containing genetic data.

        Returns:
            pd.DataFrame: Scaled genetic data.
        """
        nucleotide_to_number = {'A': 1, 'G': 2, 'C': 3, 'T': 4}
        df = df.applymap(lambda x: nucleotide_to_number.get(x, np.random.randint(1, 5)))

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df)
        return pd.DataFrame(scaled_features, columns=df.columns)

    def load_genetic_data(self, ptid):
        """
        Load and preprocess genetic data for a specific patient.

        Args:
            ptid (str): Patient ID.

        Returns:
            torch.Tensor: Preprocessed genetic data.
        """
        genetic_file_path = os.path.join(self.genetic_folder_path, f"{ptid}.csv")
        genetic_df = pd.read_csv(genetic_file_path)
        scaled_gen_df = self.preprocess_genetic_data(genetic_df)
        return torch.tensor(scaled_gen_df.values, dtype=torch.float32)

    def __len__(self):
        return len(self.tabular_frame)

    def __getitem__(self, idx):
        """
        Retrieve a data sample for the given index.

        Args:
            idx (int): Index of the data sample.

        Returns:
            dict: A dictionary containing tabular data, image data, genetic data, label, and file paths.
        """
        tabular_data = self.features[idx]
        ptid = self.tabular_frame.iloc[idx]['PTID']
        imageuid = self.tabular_frame.iloc[idx]['IMAGEUID']
        img_path = self.find_image_file(ptid, imageuid)

        if img_path:
            img = tio.ScalarImage(img_path)
            img_data = self.transform(img.data) if self.transform else img.data
            img_data = torch.tensor(img_data, dtype=torch.float32)
        else:
            img_data = torch.zeros(1, *self.transform.new_shape)

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
        """
        Find the path of the image file corresponding to the patient ID and IMAGEUID.

        Args:
            ptid (str): Patient ID.
            imageuid (int): IMAGEUID.

        Returns:
            str or None: Path to the image file, or None if not found.
        """
        pattern = f'ADNI_{ptid}_*_*_I{int(imageuid)}.nii'
        for file in os.listdir(self.img_folder):
            if fnmatch.fnmatch(file, pattern):
                return os.path.join(self.img_folder, file)
        return None

# Example usage of the dataset
# dataset = MultimodalDataset(csv_file='path_to_csv', img_folder='path_to_img_folder', genetic_folder_path='path_to_genetic_folder', transform=PreprocessTransform((128, 128, 128)))
# print(dataset[0])
