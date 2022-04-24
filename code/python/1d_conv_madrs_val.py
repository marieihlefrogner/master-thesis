from setup_1d_conv import *

global identifier
identifier = identifier.replace('Conv1D', 'Conv1D-MADRS-VALUE')

setup()

segments, labels, num_sensors, input_shape = create_segments_and_labels_madrs_val(1, segment_length, step)
X_train, X_test, y_train, y_test = train_test_split(segments, labels, test_size=0.2)

print(pd.DataFrame(y_test).describe())

if not model_path:
    if verbose:
        print('Creating model from scratch...')

    model = create_model_madrs(segment_length, num_sensors, input_shape, dropout=dropout)

    callbacks = [
        EarlyStopping(monitor='mean_squared_error', patience=2),
    ]

    history = train(model, X_train, y_train, batch_size, epochs, callbacks) # validation_data=(X_test, y_test))
else:
    if verbose:
        print(f'Loading model from {model_path}...')

    model = load_model(model_path)

loss, acc = evaluate(model, X_test, y_test, verbose=verbose)
print(pd.DataFrame(model.predict(X_test)).describe())
# max_y_test, max_y_pred_test = predict(model, X_test, y_test, verbose=verbose)

if not model_path:
    model.save(f'../models/{identifier}.h5')

cleanup()