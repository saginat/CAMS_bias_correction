def create_dataset(final_tensor,df_categories, time_to_predict, lookback_num,
                   without_correction=False, batch_size=32,time_split=False,verbose=False):
    
    tensor = (final_tensor[:,:1,:,:] if without_correction else final_tensor[:,1:,:,:])
    df = df_categories.drop('time',axis=1)

    rows_to_keep = df.loc[(df != 0).any(axis=1)].index

    filtered_input_tensor = tensor[rows_to_keep]

    filtered_target_dataframe = (
                                 df_categories.loc[rows_to_keep]
                                 .pipe(lambda d:d.assign(time = pd.to_datetime(d.time)))
                                 .reset_index(drop=True)
                                )

    filtered_target_dataframe = filtered_target_dataframe.astype({col: 'int' for col in df.columns if col != 'time'})


    label_tensor = torch.tensor(filtered_target_dataframe.drop('time',axis=1).values, dtype=torch.float32)

    original_dataset = CustomDataset(filtered_input_tensor,filtered_target_dataframe,label_tensor,time_to_predict=time_to_predict, lookback_num=lookback_num)


    filtered_data_X_y = []
    filtered_data_time = []

    # Filter and split the data
    for X, y, time in original_dataset:
        if (not torch.all(X.eq(-999))) & (not torch.all(y.eq(0))):
            filtered_data_X_y.append((X, y))
            filtered_data_time.append(time)
    new_dataset =  SequentialDataset(filtered_data_X_y)

    
    if time_split:
        if verbose:
            print('time_split')
        train_dataset, validation_dataset, test_dataset, train_indices, validation_indices, test_indices = split_dataset_by_size(filtered_data_X_y, train_percentage=0.7, validation_percentage=0.1, test_percentage=0.2)
        train_dataset, validation_dataset, test_dataset = SequentialDataset(train_dataset), SequentialDataset(validation_dataset), SequentialDataset(test_dataset)

        train_dataset = filter_zero_labels(train_dataset)
        val_dataset = filter_zero_labels(validation_dataset)
        test_dataset = filter_zero_labels(test_dataset)
    else:
        if verbose:
            print('random_split')
        train_dataset, test_dataset, train_indices, test_indices = train_test_split(new_dataset,filtered_data_time, test_size=0.2, random_state=42)
        train_dataset, validation_dataset , train_indices, validation_indices = train_test_split(train_dataset,train_indices, test_size=0.1, random_state=42)

  # Set your desired batch size
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validationloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_shape = (lookback_num+1, 65, 67)
    
    return trainloader, validationloader, testloader, input_shape ,train_indices, validation_indices, test_indices, filtered_data_time