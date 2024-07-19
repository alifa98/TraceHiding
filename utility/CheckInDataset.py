import math
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


class HexagonCheckInUserDataset(Dataset):
    def __init__(self, csv_file, trajectory_col_name="higher_order_trajectory", user_id_col_name='user_id', cell_to_id=None, split_ratio=0.8, random_state=42, mininum_trajectory_count=10):
        """
        This class is a PyTorch Dataset class that is used to load the dataset of trajectories and user IDs.

        Args:
            csv_file (str, mandatory): The path to the CSV file that contains the dataset.
            trajectory_col_name (str, optional): The name of the column in the CSV file that contains the trajectories. Defaults to "higher_order_trajectory".
            user_id_col_name (str, optional): The name of the column in the CSV file that contains the user IDs. Defaults to 'user_id'.
            cell_to_id (dict, optional): A dictionary that maps cell names to cell IDs. Defaults to None.
            split_ratio (float, optional): The ratio of the dataset that will be used for training. Defaults to 0.8.
            random_state (int, optional): The random state for splitting the dataset. Defaults to 42.
        """
        self.data = pd.read_csv(csv_file)
        
        # filter out the rows that has low count in trajectories
        self.data = self.data.groupby(user_id_col_name).filter(lambda x: len(x) >= mininum_trajectory_count)

        # Get all cell names
        traj_values = self.data[trajectory_col_name]
        cell_names = set()
        for trajectory in traj_values:
            for cell_name in trajectory.strip().split(sep=' '):
                if cell_name:
                    cell_names.add(cell_name)

        # Reassign user_ids to be in the range of 0 to number of users
        self.data[user_id_col_name] = pd.Categorical(
            self.data[user_id_col_name]).codes

        self.vocab_size = len(cell_names)
        self.users_size = len(self.data[user_id_col_name].unique())

        self.TRAJECTORY_COLUMN = 'cell_id_#generated'

        # Assign unique IDs to each cell name (we use id 0 for padding so we start from 1)
        if cell_to_id is None:
            self.cell_to_id = {cell_name: i+1 for i,
                               cell_name in enumerate(cell_names)}
            self.id_to_cell = {i+1: cell_name for i,
                               cell_name in enumerate(cell_names)}
        else:
            self.cell_to_id = cell_to_id
            self.id_to_cell = {
                i: cell_name for cell_name, i in cell_to_id.items()}

        # Convert the trajectory sequences into lists of cell IDs
        self.data[self.TRAJECTORY_COLUMN] = self.data[trajectory_col_name].apply(
            lambda traj: [self.cell_to_id[cell_name] for cell_name in traj.strip().split(sep=' ')])

        # Split the dataset into training and validation and training datasets in startified manner
        train_df, test_df = train_test_split(
            self.data, test_size=1-split_ratio, stratify=self.data[user_id_col_name], random_state=random_state)
        
        self.data = list(zip(train_df[self.TRAJECTORY_COLUMN], train_df[user_id_col_name]))
        self.test_data = list(zip(test_df[self.TRAJECTORY_COLUMN], test_df[user_id_col_name]))
        
        self.start = 0
        self.end = len(train_df)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # return the trajectory and the corresponding user ID
        # returning format: (trajectory, user_id)
        # trajectory is a list of cell IDs
        return self.data[index]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = 0
            iter_end = self.__len__()
        else:  # in a worker process
            # split workload
            # FIXME: set the start and end on a new overload of __init__ that takes a start and end
            per_worker = int(
                math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        return iter(self.data[iter_start: iter_end])

    def get_test_data(self):
        return self.test_data
    
    def get_train_data(self):
        return self.data