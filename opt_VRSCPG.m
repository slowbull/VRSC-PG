%--------------% 
%author zhouyuan huo
%email zhouyuan.huo@pitt.edu
%--------------% 

function [ ] = opt_VRSCPG(data, config)
%   Detailed explanation goes here
lr = config.lr;
max_epochs = config.max_epochs;
max_iters = config.max_iters;

w = zeros(1, size(data,2));
w_t = w;
timer = 0;
fprintf('iter,  obj, l2_grad,  timer f\n');

for epoch = 1:max_epochs
	tic;
	w_t = w;
	[G_t, G_dev_t,full_grad] = GD(data, w_t);
	for iter = 1:max_iters
		v = SVRG(data, w, G_t, G_dev_t, w_t, full_grad, config);
		w = w - lr * v;
		if config.l1 ~= 0
			w = sign(w).* max(0, abs(w)-lr*config.l1);
		end
	end
	cur_timer = toc;
	timer = timer + cur_timer;
	[obj, l2] = compute_obj(data,w,config);
	fprintf('%d    %.10f   %.10f   %f\n', epoch, obj, l2, timer);
end

end

function out = SVRG(data, w, G_t, G_dev_t, w_t, full_grad, config)
    dim = length(w);
    n = size(data,1);
	G = G_t;
    indexes = randperm(n);
    indexes = indexes(1:config.A);

    %% compute G 
	for i = 1:config.A
		sample_G = indexes(i);
		G_i = w;
		G_i(dim+1) = data(sample_G,:) * w'; 

		G_i_t = w_t; 
		G_i_t(dim+1) = data(sample_G,:) * w_t';
		
		G = G - (G_i_t - G_i) ./ config.A;
	end

    %% compute G'
	G_dev= G_dev_t;
    indexes = randperm(n);
    indexes = indexes(1:config.B);

	for i = 1:config.B
		sample_G = indexes(i);
		G_dev_i = diag(ones(dim,1)); 
		G_dev_i(dim+1,:) = data(sample_G,:);

		G_dev_i_t = diag(ones(dim,1)); 
		G_dev_i_t(dim+1,:) = data(sample_G,:);

		G_dev = G_dev - (G_dev_i_t - G_dev_i) ./ config.B;
	end

    %% Compute F'
    sample_F = randi(n);
    F_dev_i = data(sample_F,:);
    F_dev_i(dim+1) = -1;
    F_dev_i = F_dev_i * 2 * (data(sample_F,:) * G(1:dim)' - G(dim+1));
    F_dev_i(dim+1) = F_dev_i(dim+1) - 1;

    F_dev_i_t = data(sample_F,:);
    F_dev_i_t(dim+1) = -1;
    F_dev_i_t = F_dev_i_t * 2 * (data(sample_F,:) * G_t(1:dim)' - G_t(dim+1));
    F_dev_i_t(dim+1) = F_dev_i_t(dim+1) - 1;
	
    %% update value: G' * F'
    grad = F_dev_i * G_dev;
    grad_t = F_dev_i_t * G_dev_t;

	out = grad - grad_t + full_grad;
end

function [G, G_dev, grad] = GD(data, w)
    dim = length(w);
    n = size(data,1);
    %% compute G 
    G_mat = repmat(w, size(data,1),1);
    G_mat(:,dim+1) = data * w'; 
    G = mean(G_mat);
    %% compute G'
    G_dev = diag(ones(dim,1)); 
    G_dev(dim+1,:) = mean(data);
    %% Compute F'
    F_dev = zeros(1,dim+1);
    for i = 1:n
        F_dev_i = data(i,:);
        F_dev_i(dim+1) = -1;
        F_dev_i = F_dev_i * 2 * (data(i,:) * G(1:dim)' - G(dim+1));
        F_dev_i(dim+1) = F_dev_i(dim+1) - 1;
        F_dev = F_dev + F_dev_i;
    end
    F_dev = F_dev / n;
    %% update value: G' * F'
    grad = F_dev * G_dev;
end
