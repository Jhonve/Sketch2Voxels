require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'optim'
require 'paths'

opt = {
  lr = 0.00005,
  lr_decay = true,
  betal = 0.5,
  batchSize = 20,
  niter = 1200,
  gpu = 2,
  name = 'test3_step2',
  cache_dir = './traindata/cache/',
  testOnly = false,
  checkpointd = './traindata/checkpoints/',
  checkpointn = 0
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end 
  
if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
  require 'cutorch'
  cutorch.setDevice(opt.gpu)
  nn.DataParallelTable.deserializeNGPUs = 1
end
  
-- Initialize data loader --
local DataLoader = paths.dofile('data_step2.lua')
print('Loading all models into memory...')
local data = DataLoader.new(opt)
print('data size: ' .. data:size())
----------------------------
  
net = paths.dofile('net_step2.lua')
  
opt.checkpointf = 'step2'
  
print(net)
  
optimState = {
  learningRate = opt.lr,
  beta1 = opt.beta1,
}
if opt.checkpointn > 0 then
  CheckFile = opt.name .. '_' .. opt.checkpointn .. '_net.t7'
  optimStateFile = opt.name .. '_' .. opt.checkpointn .. '_net_optimState.t7'
  net = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf, CheckFile))
  optimState = torch.load(paths.concat(opt.checkpointd .. opt.checkpointf, optimStateFile))
end
  
local criterion = nn.MSECriterion()
local input = torch.Tensor(opt.batchSize, 6, 256, 256)
local err
local avgErr = nil
local prevAvgErr = 1
local prevLREpoch = -1
if opt.gpu > 0 then
  input = input:cuda()
  criterion = criterion:cuda()
  net = net:cuda()
  net = cudnn.convert(net, cudnn)
end
  
local parameters, gradParameters = net:getParameters()
-- update step Adam optim
local fStep_2_x = function(x)
  net:zeroGradParameters()
  local sample, target = data:getBatch(opt.batchSize)
  local actualBatchSize = sample:size(1)
  input[{{1,actualBatchSize}}]:copy(sample)
  local output = net:forward(input[{{1,actualBatchSize}}])
  err = criterion:forward(output, target)
  local df_dz = criterion:backward(output, target)
  net:backward(input, criterion.gradInput)
  
  return err, gradParameters
end
  
begin_epoch = opt.checkpointn + 1
for epoch = begin_epoch, opt.niter do
  data:resetAndShuffle()

  if opt.lr_decay and epoch > begin_epoch and prevAvgErr - avgErr < 0.005 and (epoch - prevLREpoch) > 15 and optimState.learningRate > 1e-9 then
    optimState.learningRate = optimState.learningRate / 2
    prevLREpoch = epoch
  end

  if avgErr ~= nil then
    prevAvgErr = avgErr
  end

  avgErr = 0
  for i = 1, data:size(), opt.batchSize do
    print(('Optimizing network, learning rate: %.9f'):format(optimState.learningRate))
    optim.adam(fStep_2_x, parameters, optimState)
    local ind_low = i
    local ind_high = math.min(data:size(), i + opt.batchSize - 1)
    avgErr = avgErr + ((ind_high - ind_low + 1) * err)
    -- logging
    print(('Epoch: [%d][%8d / %8d]\t Err: %.6f'):format(epoch, (i-1)/(opt.batchSize), math.floor(data:size()/(opt.batchSize)), err * 10000))
  end

  avgErr = avgErr / data:size()
  if paths.dir(opt.checkpointd .. opt.checkpointf) == nil then
    paths.mkdir(opt.checkpointd .. opt.checkpointf)
  end
    
  if epoch % 3 == 0 then
    CheckFile = opt.name .. '_' .. epoch .. '_net.t7'
    optimStateFile = opt.name .. '_' .. epoch .. '_net_optimState.t7'
    torch.save(paths.concat(opt.checkpointd .. opt.checkpointf, CheckFile), net:clearState())
    torch.save(paths.concat(opt.checkpointd .. opt.checkpointf, optimStateFile), optimState)
  end
    
  parameters, gradParameters = nil,nil
  parameters, gradParameters = net:getParameters()
  print(('End of epoch %d / %d'):format(epoch, opt.niter))
end
