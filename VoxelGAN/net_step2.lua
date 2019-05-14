local nn = require 'nn'
require 'cunn'
require 'cudnn'

local Convolution = cudnn.SpatialConvolution
local Max = nn.SpatialMaxPooling
local ReLU = cudnn.ReLU
local LRN = nn.SpatialCrossMapLRN

local model = nn.Sequential()

-- 6 * 256 * 256 to 96 * 32 * 32
model:add(Convolution(6, 96, 11, 11, 4, 4, 5, 5))
model:add(ReLU(true))
model:add(Max(3, 3, 2, 2, 1, 1))
model:add(LRN(5))

-- 96 * 32 * 32 to 256 * 16 * 16
model:add(Convolution(96, 256, 5, 5, 1, 1, 2, 2))
model:add(ReLU(true))
model:add(Max(3, 3, 2, 2, 1, 1))
model:add(LRN(5))

-- 256 * 16 * 16 to 384 * 16 * 16
model:add(Convolution(256, 384, 3, 3, 1, 1, 1, 1))
model:add(ReLU(true))
model:add(Convolution(384, 384, 3, 3, 1, 1, 1, 1))
model:add(ReLU(true))

-- 384 * 16 * 16 to 256 * 8 * 8
model:add(Convolution(384, 256, 3, 3, 1, 1, 1, 1))
model:add(ReLU(true))
model:add(Max(3, 3, 2, 2, 1, 1))

-- F C
model:add(nn.View(16384):setNumInputDims(3))
model:add(nn.Linear(16384, 4096))
model:add(ReLU(true))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(4096, 4096))
model:add(ReLU(true))
model:add(nn.Dropout(0.5))
model:add(nn.Linear(4096, 200))

-- 200x1x1x1 -> 256x4x4x4
model:add(nn.View(200, 1, 1, 1):setNumInputDims(1))
model:add(nn.VolumetricFullConvolution(200, 256, 4, 4, 4))
model:add(nn.VolumetricBatchNormalization(256))
model:add(nn.ReLU())
-- 256x4x4x4 -> 128x8x8x8
model:add(nn.VolumetricFullConvolution(256, 128, 4, 4, 4, 2, 2, 2, 1, 1, 1))
model:add(nn.VolumetricBatchNormalization(128))
model:add(nn.ReLU())
-- 128x8x8x8 -> 64x16x16x16
model:add(nn.VolumetricFullConvolution(128, 64, 4, 4, 4, 2, 2, 2, 1, 1, 1))
model:add(nn.VolumetricBatchNormalization(64))
model:add(nn.ReLU())
-- 64x16x16x16 -> 1x32x32x32
model:add(nn.VolumetricFullConvolution(64, 1, 4, 4, 4, 2, 2, 2, 1, 1, 1))
model:add(nn.Sigmoid())

local function ConvInit(name)
    for k, v in pairs(model:findModules(name)) do
        local n = v.kW * v.kH * v.nOutputPlane
        v.weight:normal(0, math.sqrt(2/n))
        if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
        else
            v.bias:zero()
        end
    end
end
local function BNInit(name)
    for k, v in pairs(model:findModules(name)) do
        v.weight:fill(1)
        v.bias:zero()
    end
end
local function VConvinit(name)
    for k, v in pairs(model:findModules(name)) do
        local n = v.kW * v.kH * v.kT * v.nOutputPlane
        v.weight:normal(0, math.sqrt(2/n))
        if cudnn.version >= 4000 then
            if v.bias then
                v.bias:fill(0)
            end
        end
    end
end
 
ConvInit('cudnn.SpatialConvolution')
ConvInit('cudnn.SpatialFullConvolution')
ConvInit('nn.SpatialConvolution')
VConvinit('nn.VolumetricFullConvolution')
BNInit('fbnn.SpatialBatchNormalization')
BNInit('cudnn.SpatialBatchNormalization')
BNInit('nn.SpatialBatchNormalization')
BNInit('nn.VolumetricBatchNormalization')

for k,v in pairs(model:findModules('nn.Linear')) do
    v.bias:zero()
end

model:type('torch.CudaDoubleTensor')
 
model:get(1).gradInput = nil

return model
