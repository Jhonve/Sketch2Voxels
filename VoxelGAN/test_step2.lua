require 'nn'
require 'torch'
require 'xlua'
require 'util'
local image = require 'image'
local mat = require('matio')

opt = {
    gpu = 1,
    ckp = 'step2',
    name = 'test3_step2',
    epoch = 651,
    depth_dir = './Test_data_new/depth_img/',
    normal_dir = './Test_data_new/normal_img/',
    mat_out = './Test_data_new/voxels_out/'
  }
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end 

if opt.gpu > 0 then
    require 'cunn'
    require 'cudnn'
    require 'cutorch'
    cutorch.setDevice(opt.gpu)
end

checkpoint_path = './traindata/checkpoints/' .. opt.ckp

print('Loading network...')

-- ./traindata/checkpoints/step1/test1_1200_net.t7
net_path = paths.concat(checkpoint_path, opt.name .. '_' .. opt.epoch .. '_net.t7')
net = torch.load(net_path)

net:evaluate()

print('Setting inputs...')
input = torch.Tensor(1, 6, 256, 256)
output = torch.Tensor(1, 32, 32, 32)

if opt.gpu > 0 then
    net = net:cuda()
    net = cudnn.convert(net, cudnn)
    input = input:cuda()
    output = output:cuda()
end

cats = paths.dir(opt.depth_dir)
curindex = 0
for i, v in ipairs(cats) do
    if v ~= '.' and v ~= '..' then
        cat = cats[i]
        cat_img_path = paths.concat(opt.depth_dir, cat)
        input[{1, {1, 3}, {}, {}}] = image.load(cat_img_path)
        cat_img_path = paths.concat(opt.normal_dir, cat)
        input[{1, {4, 6}, {}, {}}] = image.load(cat_img_path)
        output = net:forward(input)
        output = output:float()
        mat.save(opt.mat_out .. cat:split('.')[1] .. '.mat', {['voxels'] = output[{1}]})
        curindex = curindex + 1
        print(curindex)
    end
end
