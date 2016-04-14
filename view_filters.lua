require 'torch'
require 'gnuplot'
require 'scalar_onehot_2'

local model = torch.load(arg[1])
clusts = {}
for i,k in pairs(arg) do 
   if i > 1 then
      clusts[i-1] = arg[i]
   end
end
model:updateVisuals(clusts)

