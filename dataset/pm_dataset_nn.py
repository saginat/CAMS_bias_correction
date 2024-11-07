from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

class PM_Dataset(Dataset):
    """
    A custom dataset class for time-series prediction, handling temporal input and target data.

    Attributes:
    -----------
    tensor_input : torch.Tensor
        Input tensor data, typically of shape (samples, time, channels, features).
    dataframe_target : pd.DataFrame
        Dataframe containing target timestamps.
    label_tensor : torch.Tensor
        Label tensor corresponding to the target data.
    time_to_predict : int
        Hours into the future to predict, must be a multiple of 3.
    lookback_num : int
        Number of time steps to look back for temporal context.
    """

    def __init__(self, tensor_input, dataframe_target, label_tensor, time_to_predict=24, lookback_num=1):
        """
        Initializes the pm_dataset class.

        Parameters:
        ----------
        tensor_input : torch.Tensor
            Input tensor data.
        dataframe_target : pd.DataFrame
            DataFrame containing timestamps.
        label_tensor : torch.Tensor
            Label tensor for the target data.
        time_to_predict : int, optional
            Prediction time horizon in hours, must be a multiple of 3 (default is 24).
        lookback_num : int, optional
            Number of time steps to include in lookback for temporal context (default is 1).
        """
        self.tensor_input = tensor_input
        self.dataframe_target = dataframe_target
        self.label_tensor = label_tensor
        
        # Ensure time_to_predict is a multiple of 3
        if time_to_predict == 0 or time_to_predict % 3 == 0:
            self.time_to_predict = time_to_predict
        else:
            raise ValueError('Can only predict hours in jumps of 3')
        
        # Ensure lookback_num is non-negative
        if lookback_num < 0:
            raise ValueError('lookback_num cannot be a negative number')
        self.lookback_num = lookback_num

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return self.tensor_input.shape[0]

    def __getitem__(self, index):
        """
        Retrieves a single sample, including input data, target data, and timestamp.

        Parameters:
        ----------
        index : int
            Index of the sample to retrieve.

        Returns:
        -------
        input_data : torch.Tensor
            Input tensor for the sample. Returns a tensor with error indicator if timestamps are inconsistent.
        target_data : torch.Tensor
            Target tensor for the sample; returns zero tensor if no corresponding future timestamp is found.
        time : pd.Timestamp
            Timestamp corresponding to the input data.
        """
        # Retrieve input data with lookback context if applicable
        if self.lookback_num > 0:
            # Ensure enough history is available for the lookback
            if index >= self.lookback_num:
                # Retrieve timestamps for lookback window
                lookback_timestamps = self.dataframe_target.loc[index - self.lookback_num:index - 1, 'time'].values
                lookback_timestamps = np.append(lookback_timestamps, self.dataframe_target.loc[index, 'time'])

                # Check consistency of timestamps
                if check_timestamps(lookback_timestamps[::-1]):
                    lookback_indices = np.arange(index - self.lookback_num, index + 1)
                    input_data = self.tensor_input[lookback_indices, :, :, :]
                    input_data = input_data.view(len(lookback_indices), input_data.shape[2], input_data.shape[3])
                else:
                    input_data = torch.full((1, 1), -999)  # Error indicator for inconsistent timestamps
            else:
                input_data = torch.full((1, 1), -999)  # Error indicator for insufficient history

        elif self.lookback_num == 0:
            input_data = self.tensor_input[index]

        # Handling Target Data
        X_timestamp = self.dataframe_target.loc[index, 'time']
        target_timestamp = X_timestamp + pd.DateOffset(hours=self.time_to_predict)
        matching_row = self.dataframe_target[self.dataframe_target['time'] == target_timestamp]

        if len(matching_row) > 0:
            matching_row_index = matching_row.index.values[0]
            target_data = self.label_tensor[matching_row_index]
        else:
            target_data = torch.full((1, self.label_tensor.shape[1]), 0)[0]  # Default zero tensor if target missing

        time = self.dataframe_target.loc[index, 'time']
        return input_data, target_data, time
