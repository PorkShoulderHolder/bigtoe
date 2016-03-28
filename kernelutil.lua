function build_hist(labels, res)
	local out = torch.Tensor(labels:size(1), res):fill(0)
	for i=1,labels:size(1) do
		if labels[i] ~= 0 then
			out[i][labels[i]] = 1
		end
	end	

	return out
end

function augment_time(data, proportion)
	if math.random() < proportion then
		local time_std_dev = 1

		local out = data:clone():fill(0)

		out[{{1,5}}] = data[{{1,5}}]
		out[{{data:size(1) - 5,data:size(1)}}] = data[{{data:size(1) - 5,data:size(1)}}]
		
		for i=6,data:size(1) - 6 do
			if data[i] ~= 0 then
				local perturb = math.floor((torch.randn(1)[1] / 2) + 0.5)
				--print(perturb)
				if i + perturb >= 1 and i + perturb <= out:size(1) then
					out[i + perturb] = data[i]
				end
			end
		end
		return out
	else
		return data
	end
end

function argmax( a )
	return torch.range(1,a:size(1))[a:eq(torch.max(a))]
end

function train_test_split(identifier)
	-- body
	local x,y,z = assert(loadfile('readjson.lua'))(identifier) -- include cluster partition identifier to specify which
	local t_split = 5.0 / 6.0
	local tv_split = 9.0 / 10.0

	x = x[{{1, x:size(1) * t_split}}]:clone()
--	y = y[{{1, y:size(1) * t_split}}]:clone()
--	z = z[{{1, z:size(1) * t_split}}]:clone()
	local training_data = x[{{1,x:size(1) * tv_split}}]:clone() --, y[{{1, y:size(1) * tv_split}}]:clone(), z[{{1, z:size(1) * tv_split}}]:clone()}
	local validation_data = x[{{x:size(1) * tv_split, x:size(1)}}]:clone() -- y[{{y:size(1) * tv_split, y:size(1)}}]:clone()	, z[{{z:size(1) * tv_split, z:size(1)}}]:clone()}

	return training_data, validation_data
end

function nearestIndex( input,i )
	--
	-- gets index of nearest nonzero entry to index i
	-- expects one dimensional input
	--
	local indices = torch.range(1,input:size(1))[input:ne(0)]
	local right = 1
	local left = -1
	while(i + right < input:size(1) and input[i + right] == 0) do
		right = right + 1
	end
	while(i + left > 0 and input[i + left] == 0) do
		left = left - 1
	end
	return (math.abs(left) > right and i < input:size(1)) and (i + right) or (i + left);
end