def get_metrics_binar(ground_truth,predicted,labels=[1,2],smallest_event_level=None,return_matrix=False):
    
    o = ground_truth.copy()
    p = predicted.copy()

    if type(o)!=np.ndarray:
        o = o.detach().cpu().numpy()
        p =  p.detach().cpu().numpy()
    if smallest_event_level is not None: #added change, label the others as 1, not quite clear why it wasnt the case

        
        o = np.where(o == 0, 0, np.where(o < smallest_event_level, 1, 2))
        p = np.where(p == 0, 0, np.where(p < smallest_event_level, 1, 2))
        
        if np.all(o == 0):
            return [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,]
    
    # print(o.shape, p.shape)
    tn, fp, fn, tp = confusion_matrix(o,p,labels=labels).ravel()
    acc = (tp+tn)/(tp+fp+tn+fn)
    if tp+fp==0:
        prc = np.nan
    else:
        prc = tp/(tp+fp)
    if tp+fn==0:
        rcl = np.nan
    else:
        rcl = tp/(tp+fn)
    if rcl==np.nan or prc==np.nan or rcl==0:
        prcrcl_ratio = np.nan
        prcrcl_avg = np.nan
    else:
        prcrcl_ratio = prc/rcl
        prcrcl_avg = (prc+rcl)/2
    if tp+fp+fn==0:
        csi = np.nan
    else:
        csi = tp/(tp+fp+fn)
    
    if return_matrix:
        return [acc, prc, rcl, prcrcl_ratio, prcrcl_avg, csi], confusion_matrix(o,p,labels=labels)

    return [acc, prc, rcl, prcrcl_ratio, prcrcl_avg, csi]       


def test_stations(net, testloader, smallest_event_level=9):
    
    net.eval()
    all_predictions = []
    all_labels = []
    all_input = []
    station_pred_label_dict = {}  # Store evaluation metrics for each station

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            predicted_labels = torch.argmax(outputs, dim=1)
            all_predictions.append(predicted_labels.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        

            # calculate confusion matrix for each batch
            for station_idx in range(num_stations):
                #creating a mask to take relevant predictions and labels
                station_predictions = predicted_labels[:,station_idx]
                station_labels = labels[:, station_idx]

                # calculate station-specific metrics
                if station_idx not in station_pred_label_dict:
                    station_pred_label_dict[station_idx] = {'predictions': [], 'labels': []}
                station_pred_label_dict[station_idx]['predictions'].extend(station_predictions.cpu().numpy())
                station_pred_label_dict[station_idx]['labels'].extend(station_labels.cpu().numpy())
        predictions_test = np.concatenate(all_predictions).reshape(-1, num_stations).flatten()
        ground_truth_test = np.concatenate(all_labels).reshape(-1, num_stations).flatten()
        
        total_metrics, cm = get_metrics_binar(ground_truth_test,predictions_test,labels=[1,2],smallest_event_level=smallest_event_level, return_matrix=True)

    stations_metrics = {}
    #  evaluation metrics for each station
    for station_idx, metrics in station_pred_label_dict.items():
        station_predictions = np.array(metrics['predictions'])
        station_labels = np.array(metrics['labels'])
        acc, prc, rcl, prcrcl_ratio, prcrcl_avg, csi = get_metrics_binar(station_labels,
                                                                         station_predictions,labels=[1,2],
                                                                         smallest_event_level=smallest_event_level)
        stations_metrics[station_idx] = {'acc':acc, 'prc':prc, 'rcl':rcl,
                                         'prcrcl_ratio':prcrcl_ratio, 'prcrcl_avg':prcrcl_avg,
                                         'csi':csi}
    return stations_metrics, total_metrics, cm
	
def get_season(month):
    if 3 <= month <= 5:
        return 'spring'
    elif 6 <= month <= 8:
        return 'summer'
    elif 9 <= month <= 11:
        return 'fall'
    else:
        return 'winter'

def testing_seasons(net, testloader, test_indices,smallest_event_level=10):
    net.eval()
    all_predictions = []
    all_labels = []
    all_input = []
    station_pred_label_dict = {}  # Store evaluation metrics for each station

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs) #batch, classes, stations
            predicted_labels = torch.argmax(outputs, dim=1)
            all_predictions.append(predicted_labels.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    predictions_test_time_station = np.concatenate(all_predictions).reshape(-1, num_stations)
    ground_truth_test_time_station = np.concatenate(all_labels).reshape(-1, num_stations)

    results_df = pd.DataFrame(index=test_indices, columns=['accuracy','precision', 'recall', 'prcrcl_ratio','prcrcl_avg', 'csi'])

    # Loop over timestamps and compute metrics
    for idx, timestamp in enumerate(test_indices):

        metrics = get_metrics_binar(ground_truth_test_time_station[idx],predictions_test_time_station[idx],labels=[1,2],smallest_event_level=smallest_event_level)
        results_df.loc[timestamp] = metrics
    results_df['index_in_timestamp_list'] = range(0, len(test_indices))
    results_df = results_df.sort_index()



    results_df.index = pd.to_datetime(results_df.index)
    results_df['year'] = results_df.index.year
    results_df['month'] = results_df.index.month
    results_df['season'] = results_df['month'].apply(get_season)
    
    return results_df, predictions_test_time_station.flatten(), ground_truth_test_time_station.flatten()
