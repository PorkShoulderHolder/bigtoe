require 'gnuplot'
require 'nn'

function draw_onehot( data, network )
	-- 
	-- draws one hot encoding
	--
	local onehot = network:batchFormat(data)
	 -- gnuplot.imshow(data)
end

function pad( data, padding )
	local out = torch.Tensor(data:size(1) + 2 * padding):fill(0)
	out[{{padding, data:size(1) + padding - 1}}] = data:clone()
	return out
end

function draw_onehot_nll( data, network )
	local input = pad(data, network.range)
	local to_draw = torch.log(network:batchProb(input))


	
	local stationary = {'stationary',to_draw[{{},1}], '-'}
	local  stat_auto = {'transport', to_draw[{{},2}], '-'}
	local walking = {'motile', to_draw[{{},3}], '-'}
--	local running = {'running', to_draw[{{},4}], '-'}
	local cycling = {'cycling', to_draw[{{},4}], '-'}

	gnuplot.plot(stationary, stat_auto, walking, cycling)
end