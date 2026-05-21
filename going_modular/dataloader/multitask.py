import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import os
import pandas as pd
import cv2
from typing import List, Tuple

class PhotometricDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, type_mode='albedo'):
        if not os.path.exists(csv_file):
            alt_csv = os.path.join(os.path.dirname(csv_file), 'dataset', os.path.basename(csv_file))
            if os.path.exists(alt_csv): csv_file = alt_csv
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.type_mode = type_mode
        self.file_map = {
            'albedo': 'albedo_map_new_crop.exr.npy',
            'normalmap': 'normal_map_new_crop.exr.npy',
            'depthmap': 'depth_map_new_crop.exr.npy',
        
        }
        self.unique_ids = sorted(self.df['id'].unique())
        self.id_to_label = {id_val: i for i, id_val in enumerate(self.unique_ids)}
        self.labels_list = [self.id_to_label[row['id']] for _, row in self.df.iterrows()]
        self.weightclass = {}

    def __len__(self): return len(self.df)
    def get_labels(self): return self.labels_list

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_val = row['id']
        session_name = row['session']
        file_suffix = self.file_map.get(self.type_mode, 'albedo_map_new_crop.exr.npy')
        file_path = os.path.join(self.root_dir, str(id_val), str(session_name), file_suffix)
        try:
            image_data = np.load(file_path)
            if image_data.ndim == 3 and image_data.shape[0] == 3:
                image_data = image_data.transpose(1, 2, 0)
            elif image_data.ndim == 3 and image_data.shape[2] == 1:
                image_data = np.repeat(image_data, 3, axis=-1)
            image_data = image_data.astype(np.float32)
        except:
            image_data = np.zeros((112, 112, 3), dtype=np.float32)

        label_id = self.id_to_label[id_val]
        def get_lbl(key): return int(row.get(key, 0))
        labels = torch.tensor([label_id, get_lbl('Gender'), get_lbl('Spectacles'), get_lbl('Facial_Hair'), get_lbl('Pose'), get_lbl('Emotion')], dtype=torch.long)

        if self.transform:
            res = self.transform(image=image_data)
            image_data = res['image']

        if isinstance(image_data, np.ndarray):
            X = torch.from_numpy(image_data).permute(2, 0, 1)
        else: X = image_data
        return X, labels


class ConcatCustomExrDatasetV2(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        if not os.path.exists(csv_file):
            alt_csv = os.path.join(os.path.dirname(csv_file), 'dataset', os.path.basename(csv_file))
            if os.path.exists(alt_csv): csv_file = alt_csv
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

        # Map filenames
        self.file_map = {
            'albedo': 'albedo_map_new_crop.exr.npy',
            'normal': 'normal_map_new_crop.exr.npy'
        }

        self.unique_ids = sorted(self.df['id'].unique())
        self.id_to_label = {id_val: i for i, id_val in enumerate(self.unique_ids)}
        self.labels_list = [self.id_to_label[row['id']] for _, row in self.df.iterrows()]

    def __len__(self): return len(self.df)
    def get_labels(self): return self.labels_list

    def __load_npy(self, path):
        try:
            if not os.path.exists(path): return None
            img = np.load(path)
            if img.ndim == 3 and img.shape[0] == 3: img = img.transpose(1, 2, 0)
            return img.astype(np.float32)
        except: return None

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_val = row['id']
        session = str(row['session'])

        p1 = os.path.join(self.root_dir, str(id_val), session, self.file_map['albedo'])
        p2 = os.path.join(self.root_dir, str(id_val), session, self.file_map['normal'])

        i1 = self.__load_npy(p1)
        i2 = self.__load_npy(p2)

        if i1 is None: i1 = np.zeros((112, 112, 3), dtype=np.float32)
        if i2 is None: i2 = np.zeros((112, 112, 3), dtype=np.float32) #

        
        if i2.shape[:2] != i1.shape[:2]: i2 = cv2.resize(i2, (i1.shape[1], i1.shape[0]))

        label_id = self.id_to_label[id_val]
        def get_lbl(key): return int(row.get(key, 0))
        labels = torch.tensor([label_id, get_lbl('Gender'), get_lbl('Spectacles'), get_lbl('Facial_Hair'), get_lbl('Pose'), get_lbl('Emotion')], dtype=torch.long)

        if self.transform:
            res = self.transform(image=i1, image2=i2)
            i1, i2 = res['image'], res['image2']

        to_ts = lambda x: torch.from_numpy(x).permute(2, 0, 1) if isinstance(x, np.ndarray) else x

   
        X = torch.stack((to_ts(i1), to_ts(i2)), dim=0)
        return X, labels

# --- 2. CUSTOM SAMPLER: ĐẢM BẢO UNIQUE ID TRONG BATCH ---
class UniqueIdBatchSampler(Sampler):
    """
    Sampler này đảm bảo trong mỗi batch, mỗi ID chỉ xuất hiện tối đa 1 lần.
    Giúp cân bằng việc lấy mẫu giữa các ID có nhiều ảnh (148) và ít ảnh (5).
    """
    def __init__(self, labels: List[int], batch_size: int):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.unique_labels = list(set(labels))
        
        # Tạo map: Label -> Danh sách các index của ảnh thuộc label đó
        self.label_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.label_indices:
                self.label_indices[label] = []
            self.label_indices[label].append(idx)

        # Kiểm tra batch size
        if self.batch_size > len(self.unique_labels):
            raise ValueError(f"Batch size ({self.batch_size}) lớn hơn tổng số ID ({len(self.unique_labels)}). Không thể tạo unique batch.")

    def __iter__(self):
        n_batches = len(self.labels) // self.batch_size
        for _ in range(n_batches):
            batch_labels = np.random.choice(self.unique_labels, size=self.batch_size, replace=False)
            
            batch_indices = []
            for label in batch_labels:
                # Bước 2: Với mỗi ID đã chọn, lấy ngẫu nhiên 1 ảnh thuộc ID đó
                # Cách này giúp ID có 148 ảnh và ID có 5 ảnh đều có cơ hội xuất hiện ngang nhau
                img_idx = np.random.choice(self.label_indices[label])
                batch_indices.append(img_idx)
            
            yield batch_indices

    def __len__(self):
        return len(self.labels) // self.batch_size

# --- 3. HÀM TẠO DATALOADER ---
def create_multitask_datafetcher(config, train_transform, test_transform) -> Tuple[DataLoader, DataLoader, dict]:
    dataset_dir = config['dataset_dir']
    train_csv = os.path.join(dataset_dir, 'train_split.csv')
    test_csv = os.path.join(dataset_dir, 'probe_split.csv')
    batch_size = config['batch_size']

    type_mode = config.get('type', 'albedo')
    print(f"--- Khởi tạo Single Task: {type_mode} ---")

    train_dataset = PhotometricDataset(train_csv, dataset_dir, train_transform, type_mode)
    test_dataset = PhotometricDataset(test_csv, dataset_dir, test_transform, type_mode)

    print(f"-> Train: {len(train_dataset)} ảnh | Test: {len(test_dataset)} ảnh")

    # --- CẤU HÌNH SAMPLER CHO TRAIN ---
    # Sử dụng UniqueIdBatchSampler cho tập Train để tránh trùng ID trong batch
    train_sampler = UniqueIdBatchSampler(train_dataset.get_labels(), batch_size)

    # Lưu ý: Khi dùng batch_sampler, tham số batch_size trong DataLoader phải bỏ qua (hoặc để 1),
    # shuffle phải là False (vì sampler đã shuffle rồi).
    train_dl = DataLoader(
        train_dataset, 
        batch_sampler=train_sampler, # Dùng custom sampler ở đây
        num_workers=2, 
        pin_memory=True
    )

    # Tập Test không cần sampler phức tạp, chỉ cần shuffle=False để đánh giá tuần tự
    test_dl = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )

    return train_dl, test_dl, train_dataset.weightclass

def create_concatv2_multitask_datafetcher(config, train_transform, test_transform):
    raise NotImplementedError("Fusion Task")