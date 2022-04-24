from setup_1d_conv import *

setup()

if madrs:
    output_classes = 4
    confusion_matrix_labels = MADRS_LABLES
    create_segments_and_labels = create_segments_and_labels_madrs
else:
    output_classes = 2
    confusion_matrix_labels = CATEGORIES

segments, labels, num_sensors, input_shape = create_segments_and_labels(1, segment_length, step, k_folds=k_folds)

if k_folds > 1:
    models = []
    best_loss = 10000
    best_acc = 0
    i = 0

    # set aside 20% for evaluation
    segments_train, segments_test, labels_train, labels_test = train_test_split(segments, labels, test_size=0.2)
    _labels_test = to_categorical(labels_test, output_classes)
    
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
    splits = skf.split(segments_train, labels_train)

    for train_indexes, val_indexes in splits:
        print(f'Fold: {i+1}/{k_folds}')

        _labels = to_categorical(labels_train, output_classes)

        X_train, X_val = segments_train[train_indexes], segments_train[val_indexes]
        y_train, y_val = _labels[train_indexes], _labels[val_indexes]

        model = create_model(segment_length, num_sensors, input_shape, output_classes=output_classes, dropout=dropout, verbose=0)
        history = train(model, X_train, y_train, batch_size, epochs, callbacks=[], validation_data=(X_val, y_val), verbose=1)
        loss, acc = evaluate(model, segments_test, _labels_test, verbose=1)
        
        models.append((model, history, (loss, acc)))
        
        i+=1

    scores = [x[2] for x in models]
    df = pd.DataFrame(scores, columns=['loss', 'acc'])

    df.to_csv(f'../logs/result_{k_folds}_folds_{identifier}.csv')

else:
    X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2)

    if not model_path:
        if verbose:
            print('Creating model from scratch...')

        model = create_model(segment_length, num_sensors, input_shape, output_classes=output_classes, dropout=dropout)

        if early_stop:
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=early_stop),
            ]
        else:
            callbacks = []

        history = train(model, X_train, y_train, batch_size, epochs, callbacks, validation_split=0.4)
    else:
        if verbose:
            print(f'Loading model from {model_path}...')

        model = load_model(model_path)

    loss, acc = evaluate(model, X_test, y_test, verbose=verbose)
    max_y_test, max_y_pred_test = predict(model, X_test, y_test, verbose=verbose)

    if not model_path:
        model.save(f'../models/{identifier}.h5')

        make_confusion_matrix(max_y_test, max_y_pred_test, 
                                output_file=f'../img/confusion_matrix/{identifier}.png', 
                                print_stdout=True, 
                                xticklabels=confusion_matrix_labels, 
                                yticklabels=confusion_matrix_labels)
    else:
        make_confusion_matrix(max_y_test, max_y_pred_test, print_stdout=True)

cleanup()