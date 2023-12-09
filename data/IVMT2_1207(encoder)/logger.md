proud-valley-19: directly trained on raw model without pre-trained encoder. 5 combination data loader without noise.

wobbly-wind-24: use eager-sweep-1 pretrained encoder. 5 combination without gaussian noise


    eval (continue):
    mean 0.092225461960861
    mean1 0.09222243602959053

    validation (discrete):
    mean 0.04998719352192484
    mean1 0.04998947120059343

blooming-glade-30: use eager-sweep-1, 5 combinations without gaussian noise. (50 epoch frozen encoder) worse
    

apricot-planet-33: use eager-sweep-1. 5 combinations + gaussian noise

soft-capybara-29: use eager-sweep-1. 5 combinations + gaussian noise
    
    eval (continue):
    mean 0.11777030854057348
    mean1 0.1177684751094383

    validation (discrete):
    mean 0.08069602063764238
    mean1 0.08069709728028289


azure-cherry-31: use eager-sweep-1. one dataloader flag=2  without gaussian noise
    
    unfinished: epoch 182
    eval (continue):
    mean 0.2134503829023134
    mean1 0.2134586189367722

    validation (discrete):
    mean 0.05525102640782747
    mean1 0.05526826537950987


    unfinished: epoch 316
    mean 0.1721298992366982
    mean1 0.17213250051809204

    mean 0.0478774277220575
    mean1 0.04788676417334936

eager-sweep-1: pretrained encoder.