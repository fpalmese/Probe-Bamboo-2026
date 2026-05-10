1. One folder for each device containing the pcaps. Then run the concat to have 1 pcap per device. If pcapng -> Convert to pcap!
2. Once you have 1 pcap per device, Run the "pcap_processing/parse_pcaps.py" to process all the pcaps and convert to hex, binary and dissected
3. Run the data preprocessing ("pre_processing/data_preprocess.py") to generate a single file with all the binary cols concatenated in one single binary string (may contain U) 
    -> Produces the interim_binary df and the full Hex df (with same indices)
    -> Produces the filters if indicated int he
4. Study the device entries distribution with the "pre_processing/device_analyzer.py" [OPTIONAL]
5. Balance the dataset per device (N entries per device, with balance per type of entry) using the "pre_processing/device_analyzer.py"

6. FULL Process ->
    In cross val approach
    6.1 - Split devices in train/validation/test maybe in k-fold -> In the main loop

    6.2 - Obtain pairs for each set (use the /modules/pair_generator.py with TRAIN/VAL/TEST df) [Function Ready]
    6.3 - Generate combinations for the TEST set (C combinations per group size) -> Use the /modules/device_combination_generator.py [Function Ready]
    6.4 - TRAIN 
        - Use train pairs and bin_0_df to train bamboo to output the best filters-> Use the /modules/bamboo/bamboo_functions.py (train_bamboo fun) [Function Ready - Verified]
        - Use train bin_U_df (NOT PAIRS) to train PF (ripeti a 8, 16, 32 e 64 bits) -> Use the /modules/pf_training.py (train_pf fun one call for each M) [Function Ready - Verified]
        - NO TRAINING FOR PINTOR CLUSTERING -> Only validation/test

    6.5 - VALIDATION
        - Compute ROC on validation set for BAMBOO and extract best threshold (use filters from train) -> Use the /modules/bamboo_roc_validation.py (one for all M values) [Function Ready - Verified]
        - Compute ROC on validation set for PF and extract best threshold (use indexes from train) [Function Ready - Verified]
        - Extract best params for validation set in PINTOR (param tuning for dbscan?) -> Compute ROC on this and take best threshold

    6.6 - TEST
        - Perform clustering on BAMBOO and extract metrics (use best threshold from validation)
        - Perform clustering on PF and extract metrics (use best threshold from validation)
        - Perform clustering on IE and extract metrics (use params from validatin)
