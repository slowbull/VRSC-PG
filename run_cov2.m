%--------------% 
%author zhouyuan huo
%email zhouyuan.huo@pitt.edu
%--------------% 

load data_cov_2;

config.l1 = 1e-3;

rng(1);
config.lr = 1e-3;
config.max_iters = 100; 
config.max_epochs = 50;
config.A = 5;
config.B = 5;
opt_VRSCPG(data, config);


