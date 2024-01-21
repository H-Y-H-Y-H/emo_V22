still-sweep-1: Encoder pretrain. 6 combination dataloader
scarlet-sweep-1: Encoder pretrain. 6 combination dataloader.
eternal-sweep-1: Encoder

rosy-sky-17: use still-sweep-1 architecture without pre-trained. 5 combination dataloader without noise
driven-serenity-18: (same) use still-sweep-1 architecture without pre-trained. 5 combination dataloader without noise

    valid (discrete):
    mean 0.01950665463076744
    mean1 0.015542711292044572

    eval: (continue):
    mean 0.2316106405767983
    mean1 0.23161862636355787
misty-star-21: (same) use still-sweep-1 architecture without pre-trained. 5 combination dataloader without noise


deep-darkness-22: use pretrained still-sweep-1. 5 combination dataloader without noise

    valid (discrete):
    mean 0.01693619591355672
    mean1 0.013303275309174379

    eval: (continue):
    mean 0.2471169881226562
    mean1 0.24764134923836909
iconic-serenity-23: use pretrained still-sweep-1. 5 combination dataloader + gaussian noise 0.1

    valid (discrete):
    mean 0.050804699472831484
    mean1 0.04037751220870666

    eval: (continue):
    mean 0.20730193774247127
    mean1 0.20730752891249124

hardy-river-24: use pretrained still-sweep-1. 5 combination dataloader + gaussian noise 0.05
    
    eval: (continue):
    mean 0.24039536322997923
    mean1 0.240391152051786
new method:
use the continue data to train the model. The incorrect commands in the previous prediction will affect future,
but use predicted incorrect commands is better than use Gaussian noise with correct previous commands.

bright-darkness-36: use still-sweep-1 as pre-trained encoder. 3 combination. without noise. sequence 
hard to converge.

    mean 0.06931464800465885
    mean1 0.05917657827255526

    mean 0.21664176584988348
    mean1 0.21664528824503054

warm-water-37:use scarlet-sweep-1 as pre-trained encoder. 3 combination. without noise. sequence 

    eval
    mean 0.20338938411001217
    mean1 0.20337632051654034

upbeat-thunder-42: use still-sweep-1 as pre-trained encoder. one combination (102). without noise. 3 sequence 
   
     eval:
    mean 0.2106786909595062
    mean1 0.2108662948939009


fast-armadillo-40: use untrianed scarlet-sweep-1. model1209 Swap the encoder input and decoder input


graceful-leaf-49: use untrained still-sweep-1, one combination (100). 4 seq. deactivate ReLU

olive-plant-43: pretrained encoder: scarlet-sweep-1. train 4 combinations. valid 102 comb

fiery-sun-44: pretrained encoder: scarlet-sweep-1. train 4 comb. valid: 3 comb

snowy-silence-47: use upbeat-thunder-42 as pre-trained model. only trained 1 comb, 100, use 4 seq


lucky-sweep-7: no pretrained. train on one comb. [100].

    eval:
    mean 0.07512711579004357
    mean1 0.07240616512681568
    valid：
    mean 0.06486439827752731
    mean1 0.06030323612417133


true-sweep-2(same): no pretrained. train on one comb. [100]. use 4 seq

    eval:
    mean 0.07024393855390472
    mean1 0.06741352005577418

    valid:
    mean 0.057034405649540786
    mean1 0.05147983775780436

charmed-sky-46: use vivid-sweep-2 as pretrained model. one comb. frozen first 20 epochs.

    eval:
    mean 0.06407967790498642
    mean1 0.06107453021719776


pious-voice-56: use pretrained charmed-sky-46. 3 comb. no frozen

    eval:
    mean 0.06766361254830874
    mean1 0.06511346087135689

breezy-water-73: use pretrained charmed-sky-46. 1 comb[100]. no frozen. 7 seq loop. compare 
    
    eval:
    mean 0.061858219755144654
    mean1 0.058758071415048664

logical-rain-76: use pretrained charmed-sky-46. 1 comb[100]. no frozen. 7 seq loop. 
use k0 k1 k2 lips specific loss function.

    eval
    mean 0.06321716484803482
    mean1 0.06001206942920985

prime-sponge-77: use true-sweep-2 as pretrained model. use k0 k1 k2 lips specific loss function.
    
    eval:
    mean 0.06761959916444074
    mean1 0.06466988101361436

denim-dawn-82: use true-sweep-2 as pretrained model, but 2 comb. 100 and 102, 
the 102 doesn't join the validation. 3 seq loop. use k0 k1 k2 lips specific loss function.


