require 'categorical'

function train( data, )
	
	for i=1, data:size(1) do
		


	end

	
end

function do_epoch(data, epoch, max_epoch)
	print("epoch " .. epoch)
	shuffled_i = torch.randperm(data:size(1))
end