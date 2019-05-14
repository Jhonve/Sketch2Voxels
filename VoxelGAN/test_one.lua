require 'nn'
require 'torch'
require 'xlua'
require 'util'
local image = require 'image'
local mat = require 'matio'

opt = {
	gpu = 1,
	name = 'test1',
	epoch = 630,
	depth_dir = '../TestData/0/t_depth.jpg',
	normal_dir = '../TestData/0/t_normal.jpg',
	mat_out = '../TestData/0/t_voxels.mat'
}

for k, v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end

if opt.gpu > 0 then
	require 'cunn'
	require 'cudnn'
	require 'cutorch'
	cutorch.setDevice(opt.gpu)
end

checkpoint_path = './traindata/checkpoints/'

print('Loading network......')

net_path = paths.concat(checkpoint_path, opt.name..'_'..opt.epoch..'_net.t7')
net = torch.load(net_path)

net:evaluate()

print('Setting inputs...')
input = torch.Tensor(1, 6, 256, 256)
output = torch.Tensor(1, 32, 32, 32)
test_input = torch.Tensor(1, 6, 256, 256)
test_output = torch.Tensor(1, 32, 32, 32)

if opt.gpu > 0 then
	net = net:cuda()
	net = cudnn.convert(net, cudnn)
	input = input:cuda()
    output = output:cuda()
    test_input = test_input:cuda()
    test_output = test_output:cuda()
end

input[{1, {1, 3}, {}, {}}] = image.load(opt.depth_dir)
input[{1, {4, 6}, {}, {}}] = image.load(opt.normal_dir)
output = net:forward(input)
output = output:float()
mat.save(opt.mat_out, {['voxels'] = output[{1}]})
print('Network forward success!')

function fileWrite()
    state_file = io.open('../TestData/state.txt', 'w')
    state_file:write('0\n0\n1\n0')
end

while(1)
do
    state_file = io.open('../TestData/state.txt', 'r')
    line_list = state_file:read('*a')
    if line_list == '0\n1\n0\n0' or line_list == '0\n1\n0\n0\n' then
        print('Forwarding network......')
        test_input[{1, {1, 3}, {}, {}}] = image.load('../TestData/1/t_depth.jpg')
        test_input[{1, {4, 6}, {}, {}}] = image.load('../TestData/1/t_normal.jpg')
        test_output = net:forward(test_input)
        test_output = test_output:float()
        mat.save('../TestData/1/voxels.mat', {['voxels'] = test_output[{1}]})
        print('Network forward success!')
        if pcall(fileWrite) then
            print("State Changed")
        else
            print("PermissionError")
        end
    end
    state_file:close()
end
