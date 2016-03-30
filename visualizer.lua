require 'gnuplot'
require 'nn'

function draw_onehot( data, network )
	-- 
	-- draws one hot encoding
	--
	local onehot = network:batchFormat(data)
	gnuplot.plot({'impulses', data, '+'})
end

function pad( data, padding, offset )
	local out = torch.Tensor(data:size(1) + 2 * padding):fill(0)
	out[{{offset, data:size(1) + offset}}] = data:clone()
	return out
end

function draw_onehot_nll( data, network )
	local offset = network.see_future and network.range or 2*network.range
	local input = pad(data, network.range, offset)
	local to_draw = network:batchProb(input)[{{offset,input:size(1)}}]


	
	local stationary = {'stationary',to_draw[{{},1}], '-'}
	local  stat_auto = {'transport', to_draw[{{},2}] + to_draw[{{},3}], '-'}
	local walking = {'motile', to_draw[{{},4}] + to_draw[{{},5}] + to_draw[{{},6}], '-'}
	-- local running = {'running', , '-'}
	-- local cycling = {'cycling', to_draw[{{},6}], '-'}

	gnuplot.plot(stationary, stat_auto, walking)
end