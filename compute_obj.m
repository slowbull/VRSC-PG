
function [F, grad_l2] = compute_obj(data, w, config)
    dim = length(w);
    n = size(data,1);
    %% compute G 
    G_mat = repmat(w, size(data,1),1);
    G_mat(:,dim+1) = data * w'; 
    G = mean(G_mat);
    %% compute F
    F_mat = -ones(n,1)*G(dim+1) + (data*G(1:dim)' - ones(n,1)*G(dim+1)).^2;
    F = mean(F_mat);

	if config.l1 ~= 0
		F = F + config.l1 * sum(abs(w));
    end
    
    grad = GD(data, w);
    grad = grad + sign(w)*config.l1;
    grad = grad(:);
    grad_l2 = sum(grad.^2);
end


function [out] = GD(data, w)
    dim = length(w);
    n = size(data,1);
    %% compute G 
    G = zeros(1, dim+1);
    for i = 1:n
        G_i = w;
        G_i(dim+1) = data(i,:) * w'; 
        G = G  + G_i ./n;
    end
    %% compute G'
    G_dev = zeros(dim+1, dim);
    for i = 1:n
        G_dev_i = diag(ones(dim,1)); 
        G_dev_i(dim+1,:) = data(i,:);
        G_dev = G_dev + G_dev_i ./ n;
    end
    %% Compute F'
    F_dev = zeros(1, dim+1);
    for i = 1:n
        F_dev_i = data(i,:);
        F_dev_i(dim+1) = -1;
        F_dev_i = F_dev_i * 2 * (data(i,:) * G(1:dim)' - G(dim+1));
        F_dev_i(dim+1) = F_dev_i(dim+1) - 1;
        F_dev = F_dev + F_dev_i ./ n;
    end
    %% update value: G' * F'
    out = F_dev * G_dev;
end

