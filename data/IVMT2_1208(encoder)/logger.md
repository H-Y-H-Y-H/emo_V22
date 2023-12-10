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


