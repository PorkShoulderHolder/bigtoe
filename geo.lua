function build_kernel(labels, res)

	local out = torch.Tensor(labels:size(1), res):fill(0)
	for i=1,lat_lngs:size(1) do
		out[i][labels[i]] = 1
	end	

	return out
end