function idx = idxvec(mask)
%IDXVEC Summary of this function goes here
%   Detailed explanation goes here
idx = reshape(1:numel(mask), size(mask));
idx = idx(mask);
end

