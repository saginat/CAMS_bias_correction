
import pandas as pd
import numpy as np
import torch
from einops import rearrange

def get_metrics_using_forcast(meta_data_cams_forcast, CAMS_tensor, test_timestamps, lead_time, threshold, ground_truth_data=None, test_loader=None,
                              num_stations=38, smallest_event_level=9, classification=True):
    
    def match_obs_times_to_forcast(test_timelist, leap_time, unit='h'):
        return [x + pd.Timedelta(leap_time, unit) for x in test_timelist]

    def loader_iterator(test_loader, matching_times_indices):
        all_labels = []
        for i, (inputs, labels) in enumerate(test_loader):
            if inputs.shape[0] != 1:
                raise ValueError('Batch size is not supported')
            if i in matching_times_indices:
                labels = labels.to('cpu')  # assuming 'device' is CPU, modify if needed
                all_labels.append(labels.numpy())
        return np.concatenate(all_labels)

    def get_indices(list1, list2):
        df1 = pd.DataFrame({'Timestamp': list1})
        df2 = pd.DataFrame({'Timestamp': list2})
        merged_df = pd.merge(df1.reset_index(), df2.reset_index(), on='Timestamp', how='inner', suffixes=('_list1', '_list2'))
        indices_list1 = merged_df['index_list1'].tolist()
        indices_list2 = merged_df['index_list2'].tolist()
        if len(indices_list1) != len(indices_list2):
            raise ValueError('Lists are not matching')
        return indices_list1, indices_list2, merged_df['Timestamp'].values

    def seasons_forcast(GT, Cams_F, timestamps_list, smallest_event_level, num_stations=38):
        def get_season(month):
            if 3 <= month <= 5:
                return 'spring'
            elif 6 <= month <= 8:
                return 'summer'
            elif 9 <= month <= 11:
                return 'fall'
            else:
                return 'winter'
        
        predictions_test_time_station = Cams_F.reshape(-1, num_stations)
        ground_truth_test_time_station = GT.reshape(-1, num_stations)
        results_df = pd.DataFrame(index=timestamps_list, columns=['accuracy', 'precision', 'recall', 'prcrcl_ratio', 'prcrcl_avg', 'csi'])
        
        for idx, timestamp in enumerate(timestamps_list):
            metrics = get_metrics_binar2(ground_truth_test_time_station[idx], predictions_test_time_station[idx], labels=[1, 2], smallest_event_level=smallest_event_level)
            results_df.loc[timestamp] = metrics
        results_df['index_in_timestamp_list'] = range(len(timestamps_list))
        results_df = results_df.sort_index()
        results_df.index = pd.to_datetime(results_df.index)
        results_df['year'] = results_df.index.year
        results_df['month'] = results_df.index.month
        results_df['season'] = results_df['month'].apply(get_season)
        return results_df

    def test_stations_cams_forcast(ground_truth, predictions, smallest_event_level=9):
        station_pred_label_dict = {}
        for station_idx in range(ground_truth.shape[1]):
            station_labels = ground_truth[:, station_idx]
            station_predictions = predictions[:, station_idx]
            station_pred_label_dict.setdefault(station_idx, {'predictions': [], 'labels': []})
            station_pred_label_dict[station_idx]['predictions'].extend(station_predictions)
            station_pred_label_dict[station_idx]['labels'].extend(station_labels)

        stations_metrics = {}
        for station_idx, metrics in station_pred_label_dict.items():
            station_predictions = np.array(metrics['predictions'])
            station_labels = np.array(metrics['labels'])
            acc, prc, rcl, prcrcl_ratio, prcrcl_avg, csi = get_metrics_binar2(station_labels, station_predictions, labels=[1, 2], smallest_event_level=smallest_event_level)
            stations_metrics[station_idx] = {'acc': acc, 'prc': prc, 'rcl': rcl, 'prcrcl_ratio': prcrcl_ratio, 'prcrcl_avg': prcrcl_avg, 'csi': csi}
        return stations_metrics

    moved_test_timestamps = match_obs_times_to_forcast(test_timestamps, lead_time)
    forcast_indices, test_indices, timestamps_lst = get_indices(meta_data_cams_forcast['timestamp'], moved_test_timestamps)
    Cams_forcast_reshaped = rearrange(CAMS_tensor, 's t l -> t s l')
    leap_index = meta_data_cams_forcast['lead_time'].index(lead_time)
    Cams_forcast_subset = Cams_forcast_reshaped[forcast_indices, :, leap_index]
    
    if test_loader is not None:
        ground_truth = loader_iterator(test_loader, test_indices)
    else:
        ground_truth = ground_truth_data[test_indices, :]

    if classification:
        Cams_forcast_subset[Cams_forcast_subset <= 0] = np.nan
        nan_mask = torch.isnan(Cams_forcast_subset)
        Cams_forcast_subset[nan_mask] = 0
        mask = (Cams_forcast_subset < threshold) & (Cams_forcast_subset > 0)
        Cams_forcast_subset = torch.where(mask, 1, Cams_forcast_subset)
        Cams_forcast_subset[Cams_forcast_subset >= threshold] = 2

    cams_numpy = Cams_forcast_subset.detach().cpu().numpy()
    total_metric, confusion_matrix = get_metrics_binar2(ground_truth.flatten(), cams_numpy.flatten(), labels=[1, 2], smallest_event_level=smallest_event_level, return_matrix=True)
    seasons_df = seasons_forcast(ground_truth, cams_numpy, timestamps_lst, smallest_event_level=smallest_event_level, num_stations=num_stations)
    seasons_df['leap_time'] = lead_time
    stations_metrics = test_stations_cams_forcast(ground_truth, cams_numpy, smallest_event_level=smallest_event_level)

    return total_metric, seasons_df, stations_metrics, ground_truth, cams_numpy, timestamps_lst, confusion_matrix

def get_cams_forecast_pred(station, meta_data=None, tensor_data=None, load_tensor_forecast_pm10=False, israel_data=False):
    try:
        import xarray as xr
    except ImportError:
        print("Please install xarray via conda or pip.")
    
    if load_tensor_forecast_pm10:
        tensor_data = torch.load("/home/labs/rudich/gavriel/pm_10/data_til_01.2015/new_data_pm10_cams.pkl")
        meta_data = torch.load("/home/labs/rudich/ronsar/CAMS_forecasts/gavriel_meta.pkl")

    list_pm = []
    lat_lon_station = {}
    meta_data_pm = {}
    tensor_lat = torch.tensor(meta_data['latitude'])
    tensor_lon = torch.tensor(meta_data['longitude'])
    
    for i in range(len(station)):
        station_id = torch.tensor(station['stn_id'].iloc[i])
        lat = torch.tensor(station['lat'].iloc[i])
        lon = torch.tensor(station['lon'].iloc[i])
        min_lat = torch.argmin(abs(tensor_lat - lat)).item()
        min_lon = torch.argmin(abs(tensor_lon - lon)).item()
        lat_lon_station[i] = [min_lat, min_lon]

    for i in range(len(station)):
        list_pm.append(tensor_data[:, :, lat_lon_station[i][0], lat_lon_station[i][1]])
    tensor3d_pm = torch.stack(list_pm)

    meta_data_pm["lead_time"] = [x for x in range(109) if x % 3 == 0]
    meta_data_pm["latitude"] = meta_data["latitude"]
    meta_data_pm["longitude"] = meta_data["longitude"]
    meta_data_pm["station_lan_lon"] = lat_lon_station
    meta_data_pm["timestamp"] = meta_data["timestamp"]
    
    if israel_data:
        meta_data_pm["station_id_name"] = station['stn_name']
    
    return tensor3d_pm, meta_data_pm
