class GlobalSettings :
    #DataPreProcessor settings
    BALL_RADIUS = 500
    PASSES = 10
    PLOTTING = False
    NORMALIZATION_PEAK = 1000
    WINDOW_LENGTH = 7
    POLYORDER = 2

    #Files settings
    CLEAN_DATA = False
    DATA_FILES_LIST_CONTROL = ["/Users/stepanpijacek/Documents/GitHub/FrequencyAnalysisDS/RawData/kvasinky s atm set/miska/kontrola",
                       "/Users/stepanpijacek/Documents/GitHub/FrequencyAnalysisDS/RawData/kvasinky s atm set/tekute/kontroly"]

    DATA_FILES_LIST_ANOMALOUS = ["/Users/stepanpijacek/Documents/GitHub/FrequencyAnalysisDS/RawData/kvasinky s atm set/miska/2023-04-13 kvasinky s atm/text",
                        "/Users/stepanpijacek/Documents/GitHub/FrequencyAnalysisDS/RawData/kvasinky s atm set/miska/2023-04-20 kvasinky s atm/text",
                        "/Users/stepanpijacek/Documents/GitHub/FrequencyAnalysisDS/RawData/kvasinky s atm set/miska/2023-05-02 kvasinky s atm/text",
                        "/Users/stepanpijacek/Documents/GitHub/FrequencyAnalysisDS/RawData/kvasinky s atm set/tekute/2022-12-06 amfotericin b/text",
                        "/Users/stepanpijacek/Documents/GitHub/FrequencyAnalysisDS/RawData/kvasinky s atm set/tekute/2022-12-08 flukonazol/text",
                        "/Users/stepanpijacek/Documents/GitHub/FrequencyAnalysisDS/RawData/kvasinky s atm set/tekute/2022-12-13 vorikonazol/text"]

    #Augmentation settings
    NOISE_LEVEL = 0.01
    DRIFT_FACTOR = 0.005
    SCALE_FACTOR = 1.2
    NUM_AUGMENTED_SAMPLES = 100

    #Model settings
    TRAIN_MODEL = True
    SHOW_ENCODER_LAYERS = False
    CONTROL_DATA_FILE_PATH = "/Users/stepanpijacek/Documents/GitHub/FrequencyAnalysisDS/CleanData/kontrol"
    ANOMALIES_DATA_FILE_PATH = "/Users/stepanpijacek/Documents/GitHub/FrequencyAnalysisDS/CleanData"
    NUM_EPOCHS = 300
    BATCH_SIZE = 59
    STEPS_PER_EPOCH = 500
    LATENT_DIM = 507
    INPUT_SIZE = 1016
    CODE_SIZE = 507
    VALIDATION_PERCENTAGE = 0.2
    VALIDATION_STEPS = 800
    LEARNING_RATE = 0.001
    ENCODER_ACTIVATION = "relu"
    DECODER_ACTIVATION = "sigmoid"
    LOSS = "mean_squared_error"
    WEIGHT_PATH = "/Users/stepanpijacek/Documents/GitHub/FrequencyAnalysisDS/Output/{}_weights.best.hdf5".format('AnomalyDetection')