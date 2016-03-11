require 'gnuplot'


function draw_onehot( data, network )
	-- 
	-- draws one hot encoding
	--
	local onehot = network:batchFormat(data)
	gnuplot.imshow(data)
end

function draw_onehot_nll( data, network )
	local input = torch.Tensor(data:size(1) + 2* network.range, data:size(2)):fill(0)
	local to_draw = network:batchProb(input)
	gnuplot.plot(input)

	local stationary = {'stationary',input[{1,{}}], '-'}
	local  stat_auto = {'stationary-automotive', input[{2,{}}], '-'}
	local auto = {'automotive', input[{3,{}}], '-'}
	local walking = {'walking', input[{4,{}}], '-'}
	local running = {'running', input[{5,{}}], '-'}
	local cycling = {'cycling', input[{6,{}}], '-'}

	gnuplot.plot(stationary, stat_auto, auto, walking, running, cycling)
end