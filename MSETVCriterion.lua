require 'torch'
require 'nn'

local MSETVCriterion, parent = torch.class('nn.MSETVCriterion', 'nn.ParallelCriterion')

function MSETVCriterion:__init(mse_coef, tv_coef, l1_coef, l2_coef)
   parent.__init(self)
   self.gradInput = {}
   self.mod_counter = 0

   self.tv_coef = tv_coef or 0
   self.l1_coef = l1_coef or 0.05

   self.l2_act_coef = 0.05
   
   self.bg_time = 0.05
   self.act_time = 0.05
   self.loc_time = 0.15

   self.mse_coef = mse_coef or 0.65
   
   self.msecrit = nn.MSECriterion()
   self.l1_crit = nn.AbsCriterion()
   self.l2_crit = nn.MSECriterion()
   self.tv_crit = nn.AbsCriterion()

   self:add(self.msecrit, self.mse_coef)
   self:add(self.l1_crit, self.l1_coef)
   self:add( nn.MSECriterion(), self.act_time)
   self:add( nn.MSECriterion(), self.loc_time)
   self:add( nn.MSECriterion(), self.bg_time)
   self:add( nn.MSECriterion(), self.l2_act_coef)
   --self:add(self.tv_crit, self.tv_coef)

end

-- function MSETVCriterion:formatInput( input, target )
--    -- body
-- end

-- function tvVariationForwards(input, p)
--    p = p or 1
--    local copy = input:clone():fill(0)
--    copy[{{1,input:size(1) - 1}}] = input[{{2,input:size(1)}}]
--    copy[input:size(1)] = input[input:size(1)]
--    return self.tv_crit:updateOutput(input,copy)
-- end

-- function tvVariationBackwards(input, p)
--    p = p or 1
--    local copy = input:clone():fill(0)
--    copy[{{1,input:size(1) - 1}}] = input[{{2,input:size(1)}}]
--    copy[input:size(1)] = input[input:size(1)]
--    return self.tv_crit:updateGradInput(input,copy)
-- end

-- function MSETVCriterion:updateOutput(input, target)
--    self.output = 0
--    local l1_zs = input[1]:clone():fill(0)
--    local l2_zs = input[2]:clone():fill(0)
--    self.output = self.output + self.mse_coef * self.msecrit(input[1], target)
--    self.output = self.output + self.l1_coef * self.l1_crit:updateOutput(input[2], l1_zs)
--   -- print(input[2], l2_zs)
--    --self.output = self.output + self.l2_coef * self.l2_crit:updateOutput(input[3], l2_zs)
--  ---  self.output = self.output + self.tv_coef * tvVariation(input[4])
--    return self.output
-- end

-- function MSETVCriterion:updateGradOutput(input, target)
--    self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
--    nn.utils.recursiveFill(self.gradInput, 0)
--    local l1_zs = input[1]:clone():fill(0)
--    local l2_zs = input[2]:clone():fill(0)
--    nn.utils.recursiveAdd(self.gradInput[1], self.mse_coef, self.msecrit:updateGradInput(input[1], target))
--    nn.utils.recursiveAdd(self.gradInput[2], self.l1_coef, self.l1_coef:updateGradInput(input[2], l1_zs))
--   -- nn.utils.recursiveAdd(self.gradInput[3], self.l2_coef, self.l2_coef:updateGradInput(input[3], l2_zs))
--  --  nn.utils.recursiveAdd(self.gradInput[4], self.tv_coef, tvVariationBackwards(input[4]))
--    return self.gradInput
-- end
