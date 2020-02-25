def return_audio_classifier(cluster_length, height, width, channel,summary=False, load_weights=True):
    
    model = Sequential()

    ## NET MODEL 0:
    #
    # INPUT -> [CONV -> RELU -> CONV -> RELU -> POLL] ->
    # -> [CONV -> RELU -> CONV -> RELU -> POLL] -> FC -> RELU -> FC
    #
    # - IMPLEMENTED METHOD-

    # First layer
    model.add(Convolution2D(n_filters_1, d_filter, d_filter, border_mode='valid', input_shape=(data_w, data_h, 3)))
    model.add(Activation('relu'))

    # Second layer
    model.add(Convolution2D(n_filters_1, d_filter, d_filter))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Drop layer
    model.add(Dropout(p_drop_1))

    # Third layer
    model.add(Convolution2D(n_filters_2, d_filter, d_filter, border_mode='valid'))
    model.add(Activation('relu'))

    # Fouth layer
    model.add(Convolution2D(n_filters_2, d_filter, d_filter))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Drop layer
    model.add(Dropout(p_drop_1))

    ## Used to flat the input (1, 10, 2, 2) -> (1, 40)
    model.add(Flatten())

    # Full Connected layer
    model.add(Dense(256))
    model.add(Activation('relu'))
    # Drop layer
    model.add(Dropout(p_drop_2))

    # Output Full Connected layer
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    if summary:
        print(model.summary())
        
    return model